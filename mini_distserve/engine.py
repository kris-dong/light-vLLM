"""
Prefill / decode engines.

A single ``Engine`` instance owns:
  * a ``ModelPartition`` (sizing for its TP/PP slice),
  * a ``KVAllocator`` for its KV blocks,
  * a ``Backend`` that performs the actual forward (mock or transformers).
  * an asyncio task loop that pulls work from its scheduler each iteration.

The ``role`` field controls which work it accepts:
  * PREFILL: takes a request, runs prompt forward, emits first token + KV.
  * DECODE:  receives KV+token from prefill, runs incremental decode steps.
  * COLOCATED: does both.

KV transfer between prefill and decode engines is done via ``kv_handoff``:
in mock mode it just copies block ids and sleeps for a modeled time; with a
real backend it would be NCCL/RDMA. The framework structure is the point.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .kv_allocator import KVAllocator, KVAllocation
from .partition import ModelPartition
from .router import EngineRole, Request


def _resolve_loadable_id_and_cache(model_path: str) -> Tuple[str, Optional[str]]:
    """
    Decide whether to load directly from a local snapshot or fall back to the
    HF repo id. Returns ``(load_id, cache_dir)`` where ``load_id`` is what gets
    passed to ``from_pretrained`` and ``cache_dir`` (if not None) tells HF
    where to download/look for shards.
    """
    import os
    import re

    p = os.path.abspath(model_path)
    has_weights = False
    if os.path.isdir(p):
        for entry in os.listdir(p):
            low = entry.lower()
            if low.endswith(".safetensors") or low.endswith(".bin"):
                has_weights = True
                break
    if has_weights:
        return p, None

    # No weights locally — try to derive the HF repo id from the cache layout.
    # Path looks like .../<cache_root>/hub/models--ORG--NAME/snapshots/<sha>/
    m = re.search(r"/hub/models--([^/]+)/snapshots/", p + "/")
    if m:
        repo_id = m.group(1).replace("--", "/")
        # cache_root is the parent of "hub/".
        cache_root = p.split("/hub/")[0]
        return repo_id, cache_root

    # Last resort: hand the path back as-is and let HF complain meaningfully.
    return p, None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Abstracts the forward pass so we can swap mock <-> real Qwen-7B."""

    async def prefill(self, request: Request) -> Tuple[int, Dict[str, Any]]:
        """Returns (first_token_id, kv_state_handle)."""
        ...

    async def decode_step(
        self, request_id: str, prev_token: int, kv_state: Dict[str, Any]
    ) -> int:
        """Runs one incremental token step. Returns next token id."""
        ...

    async def receive_kv(
        self, request_id: str, kv_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hook for decode engine to install KV transferred from prefill."""
        ...

    async def release(self, request_id: str) -> None:
        ...


class MockBackend:
    """
    Backend that uses the router's cost model to simulate timing.
    No real model, no GPU. Useful for orchestration tests.
    """

    def __init__(self, partition: ModelPartition,
                 prefill_ms_per_token: float = 0.015,
                 decode_ms_per_step: float = 0.05,
                 kv_xfer_ms_per_block: float = 0.05) -> None:
        self.partition = partition
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_step = decode_ms_per_step
        self.kv_xfer_ms_per_block = kv_xfer_ms_per_block
        self._kv: Dict[str, Dict[str, Any]] = {}

    async def prefill(self, request: Request) -> Tuple[int, Dict[str, Any]]:
        ms = request.prompt_len * self.prefill_ms_per_token
        await asyncio.sleep(ms / 1000.0)
        # "KV state" is just metadata in mock.
        kv = {"len": request.prompt_len, "next_pos": request.prompt_len}
        self._kv[request.request_id] = kv
        # Pseudo-deterministic first token.
        first_token = (sum(request.prompt_tokens[-8:]) + 7) % 32000
        return first_token, kv

    async def decode_step(
        self, request_id: str, prev_token: int, kv_state: Dict[str, Any]
    ) -> int:
        await asyncio.sleep(self.decode_ms_per_step / 1000.0)
        kv_state["next_pos"] += 1
        # Cheap deterministic walk so output is reproducible.
        return (prev_token * 1103515245 + 12345) & 0x7FFF

    async def receive_kv(
        self, request_id: str, kv_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Simulate transfer time proportional to KV size.
        n_blocks = max(1, kv_state.get("len", 1) // self.partition.block_size_tokens)
        await asyncio.sleep(n_blocks * self.kv_xfer_ms_per_block / 1000.0)
        self._kv[request_id] = dict(kv_state)
        return self._kv[request_id]

    async def release(self, request_id: str) -> None:
        self._kv.pop(request_id, None)


class TransformersBackend:
    """
    Real Qwen-7B backend via HuggingFace transformers.

    Single-GPU per engine: loads on ``device_ids[0]`` via plain ``.to(device)``
    (no ``accelerate`` dependency). Multi-GPU per engine falls back to
    ``device_map="auto"``, which requires ``pip install accelerate``. For real
    tensor parallelism, swap this backend for vLLM or TGI; the interface is
    identical to ``MockBackend`` so the rest of the framework doesn't care.
    """

    def __init__(self, partition: ModelPartition,
                 max_new_tokens_cap: int = 1024) -> None:
        # Imports are deferred so the rest of the package works without torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.partition = partition
        self.torch = torch
        device_ids = list(partition.layout.device_ids)

        dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp16": torch.float16, "float16": torch.float16,
                 "fp32": torch.float32}.get(partition.dtype, torch.bfloat16)

        load_id, cache_dir = _resolve_loadable_id_and_cache(partition.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(load_id, cache_dir=cache_dir)

        # transformers >=4.49 prefers `dtype`; earlier versions only accept
        # `torch_dtype`. Send both to stay compatible with the autoawq-pinned
        # transformers 4.51.3.
        load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        # Quantized AWQ/GPTQ models can't be moved with `.to()` after load —
        # the quant linear modules embed the destination device in their packed
        # tensors. Always pin via device_map.
        is_quantized = partition.quant_method in ("awq", "gptq")

        if len(device_ids) == 1:
            target = f"cuda:{device_ids[0]}"
            if is_quantized:
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_id, device_map={"": target}, **load_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_id, **load_kwargs
                )
                self.model.to(target)
        else:
            # Multi-GPU per engine: needs accelerate for device_map="auto".
            try:
                import accelerate  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "Multi-GPU per engine requires `accelerate`. Install with "
                    "`pip install accelerate`, or set device_ids=(N,) (one GPU "
                    "per engine) to use the plain load path."
                ) from e
            import os as _os
            _os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            self.model = AutoModelForCausalLM.from_pretrained(
                load_id, device_map="auto", **load_kwargs
            )

        self.model.eval()
        self.max_new_tokens_cap = max_new_tokens_cap
        self._kv: Dict[str, Any] = {}

    async def prefill(self, request: Request) -> Tuple[int, Dict[str, Any]]:
        torch = self.torch
        input_ids = torch.tensor(
            [list(request.prompt_tokens)], dtype=torch.long, device=self.model.device
        )
        loop = asyncio.get_event_loop()

        def _run() -> Tuple[int, Any]:
            with torch.inference_mode():
                out = self.model(input_ids=input_ids, use_cache=True)
            logits = out.logits[:, -1, :]
            tok = int(torch.argmax(logits, dim=-1).item())
            return tok, out.past_key_values

        first_token, past = await loop.run_in_executor(None, _run)
        kv = {"past": past, "len": request.prompt_len, "next_pos": request.prompt_len}
        self._kv[request.request_id] = kv
        return first_token, kv

    async def decode_step(
        self, request_id: str, prev_token: int, kv_state: Dict[str, Any]
    ) -> int:
        torch = self.torch
        loop = asyncio.get_event_loop()
        device = self.model.device
        ids = torch.tensor([[prev_token]], dtype=torch.long, device=device)
        past = kv_state["past"]

        def _run() -> Tuple[int, Any]:
            with torch.inference_mode():
                out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            tok = int(torch.argmax(logits, dim=-1).item())
            return tok, out.past_key_values

        next_token, past = await loop.run_in_executor(None, _run)
        kv_state["past"] = past
        kv_state["next_pos"] += 1
        return next_token

    async def receive_kv(
        self, request_id: str, kv_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        # KV produced on the prefill engine's GPU has to be moved to this
        # engine's GPU before decode can use it. In a real disaggregated
        # deployment this is the NCCL/RDMA hop; in-process we do an explicit
        # `.to(device)`. We mutate ``kv_state["past"]`` in place so the prefill
        # engine's eviction path doesn't see stale tensors.
        torch = self.torch
        target = self.model.device
        past = kv_state.get("past")
        if past is not None:
            kv_state["past"] = _move_past_kv(past, target, torch)
        self._kv[request_id] = kv_state
        return kv_state

    async def release(self, request_id: str) -> None:
        self._kv.pop(request_id, None)


def _move_past_kv(past, target_device, torch):
    """Move HF past_key_values to ``target_device``. Handles both the legacy
    tuple-of-tuples form and the newer ``DynamicCache`` object."""
    # DynamicCache: has .key_cache / .value_cache lists of tensors per layer.
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for i, (k, v) in enumerate(zip(past.key_cache, past.value_cache)):
            if k.device != target_device:
                past.key_cache[i] = k.to(target_device, non_blocking=True)
            if v.device != target_device:
                past.value_cache[i] = v.to(target_device, non_blocking=True)
        return past
    # Legacy: tuple of (k, v) per layer.
    moved = []
    for k, v in past:
        moved.append((k.to(target_device, non_blocking=True),
                      v.to(target_device, non_blocking=True)))
    return tuple(moved)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class _ActiveSeq:
    request: Request
    alloc: KVAllocation
    last_token: int
    kv_state: Dict[str, Any]
    tokens_produced: int = 0
    output: List[int] = field(default_factory=list)
    # Set when the request finishes.
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Engine:
    """One worker. Owns KV blocks, runs forward, reports state to the router."""

    def __init__(
        self,
        engine_id: str,
        role: EngineRole,
        partition: ModelPartition,
        backend: Backend,
        max_batch_size: int = 64,
        kv_safety_margin_blocks: int = 32,
    ) -> None:
        self.engine_id = engine_id
        self.role = role
        self.partition = partition
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.kv_safety_margin_blocks = kv_safety_margin_blocks

        self.allocator = KVAllocator(
            total_blocks=partition.total_kv_blocks,
            block_size_tokens=partition.block_size_tokens,
        )

        self._active: Dict[str, _ActiveSeq] = {}
        # Output of completed sequences, kept until the caller reads them.
        self._completed: Dict[str, List[int]] = {}
        self._completion_events: Dict[str, asyncio.Event] = {}
        # Rolling stats reported up to the router.
        self.recent_ttft_ms: float = 0.0
        self.recent_tpot_ms: float = 0.0
        self.waiting_queue_len: int = 0

    # ---- request lifecycle (called by scheduler) -------------------------------

    async def admit_prefill(self, request: Request) -> int:
        """Allocate KV for prompt, run prefill, return first token."""
        alloc = self.allocator.allocate(
            request_id=request.request_id,
            num_tokens=request.prompt_len,
            preemptible=True,
            priority=request.priority,
        )
        t0 = time.perf_counter()
        first_token, kv = await self.backend.prefill(request)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        self.recent_ttft_ms = ttft_ms

        seq = _ActiveSeq(
            request=request,
            alloc=alloc,
            last_token=first_token,
            kv_state=kv,
        )
        seq.output.append(first_token)
        seq.tokens_produced = 1
        self._active[request.request_id] = seq
        self._completion_events[request.request_id] = asyncio.Event()
        return first_token

    async def admit_decode(
        self, request: Request, first_token: int, kv_state: Dict[str, Any]
    ) -> None:
        """Decode-pool entry point: install KV transferred from prefill."""
        # Hand KV to backend (transfer + install); takes time in mock mode.
        installed = await self.backend.receive_kv(request.request_id, kv_state)
        # Reserve KV blocks locally for the prompt + first token + predicted decode.
        reserve_tokens = request.prompt_len + 1 + request.predicted_new_tokens
        alloc = self.allocator.allocate(
            request_id=request.request_id,
            num_tokens=reserve_tokens,
            preemptible=True,
            priority=request.priority,
        )
        seq = _ActiveSeq(
            request=request,
            alloc=alloc,
            last_token=first_token,
            kv_state=installed,
        )
        seq.output.append(first_token)
        seq.tokens_produced = 1
        self._active[request.request_id] = seq
        self._completion_events[request.request_id] = asyncio.Event()

    async def step(self) -> List[str]:
        """One iteration: advance every active sequence by one token. Returns finished ids."""
        if not self._active:
            return []
        finished: List[str] = []
        t0 = time.perf_counter()

        # Run all sequences in parallel for this iteration.
        async def _one(rid: str, seq: _ActiveSeq) -> Optional[str]:
            tok = await self.backend.decode_step(rid, seq.last_token, seq.kv_state)
            seq.output.append(tok)
            seq.last_token = tok
            seq.tokens_produced += 1
            # Grow KV by one token; may need a fresh block.
            try:
                self.allocator.extend(rid, extra_tokens=1)
            except RuntimeError:
                # Out of KV mid-decode — abort this seq, scheduler will spill.
                seq.done.set()
                return rid

            if seq.tokens_produced >= seq.request.max_new_tokens:
                seq.done.set()
                return rid
            return None

        results = await asyncio.gather(
            *[_one(rid, seq) for rid, seq in list(self._active.items())]
        )
        for rid in results:
            if rid is not None:
                finished.append(rid)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Average per-seq ms is a fine TPOT proxy when batching is uniform.
        n = max(1, len(self._active))
        self.recent_tpot_ms = elapsed_ms / n
        return finished

    async def reap(self, request_id: str) -> List[int]:
        """Free KV and stash the finished sequence's output for the caller to await."""
        seq = self._active.pop(request_id, None)
        if seq is None:
            return self._completed.pop(request_id, [])
        await self.backend.release(request_id)
        self.allocator.free(request_id)
        self._completed[request_id] = list(seq.output)
        ev = self._completion_events.get(request_id)
        if ev is not None:
            ev.set()
        return list(seq.output)

    async def wait_for(self, request_id: str) -> List[int]:
        """Block until the request's output is ready, then consume it."""
        ev = self._completion_events.get(request_id)
        if ev is None:
            # Already reaped or never admitted — return whatever is stashed.
            return self._completed.pop(request_id, [])
        await ev.wait()
        self._completion_events.pop(request_id, None)
        return self._completed.pop(request_id, [])

    async def evict(self, request_id: str) -> None:
        """Spill a request — drop its state and free KV. Caller will re-admit later."""
        seq = self._active.pop(request_id, None)
        if seq is not None:
            seq.done.set()
            await self.backend.release(request_id)
        self.allocator.free(request_id)
        # Wake any waiter so they don't block forever on an evicted request.
        ev = self._completion_events.pop(request_id, None)
        if ev is not None:
            ev.set()

    # ---- handoff (used by ServingSystem to bridge prefill -> decode) -----------

    async def export_kv(self, request_id: str) -> Dict[str, Any]:
        seq = self._active.get(request_id)
        if seq is None:
            raise KeyError(request_id)
        return seq.kv_state

    # ---- introspection ---------------------------------------------------------

    def active_count(self) -> int:
        return len(self._active)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "role": self.role.value,
            "active": len(self._active),
            "kv": self.allocator.snapshot(),
            "recent_ttft_ms": self.recent_ttft_ms,
            "recent_tpot_ms": self.recent_tpot_ms,
        }
