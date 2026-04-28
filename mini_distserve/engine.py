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
from collections import OrderedDict
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

    async def prefill(
        self,
        request: Request,
        cached_kv: Optional[Dict[str, Any]] = None,
        cached_tokens: int = 0,
    ) -> Tuple[int, Dict[str, Any]]:
        """Run prefill. If ``cached_kv`` is given, treat its ``past`` as the KV
        for the first ``cached_tokens`` prompt tokens and only forward the rest
        of the prompt — i.e., resume prefill from a cached prefix.
        Returns (first_token_id, kv_state_handle)."""
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

    async def snapshot_prefix(
        self,
        request_id: str,
        kv_state: Dict[str, Any],
        num_tokens: int,
    ) -> Dict[str, Any]:
        """Detached snapshot of the first ``num_tokens`` of ``kv_state`` so the
        engine's prefix cache can survive the original request finishing.
        Implementations should ``.clone()`` tensors so future request growth
        doesn't corrupt the cache."""
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

    async def prefill(
        self,
        request: Request,
        cached_kv: Optional[Dict[str, Any]] = None,
        cached_tokens: int = 0,
    ) -> Tuple[int, Dict[str, Any]]:
        # On a cache hit only forward the suffix; sleep simulates that compute.
        suffix_tokens = max(1, request.prompt_len - max(0, cached_tokens))
        ms = suffix_tokens * self.prefill_ms_per_token
        await asyncio.sleep(ms / 1000.0)
        kv = {"len": request.prompt_len, "next_pos": request.prompt_len}
        self._kv[request.request_id] = kv
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

    async def snapshot_prefix(
        self,
        request_id: str,
        kv_state: Dict[str, Any],
        num_tokens: int,
    ) -> Dict[str, Any]:
        # No tensors in mock; just record the cached length.
        return {"len": num_tokens, "next_pos": num_tokens}

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

    async def prefill(
        self,
        request: Request,
        cached_kv: Optional[Dict[str, Any]] = None,
        cached_tokens: int = 0,
    ) -> Tuple[int, Dict[str, Any]]:
        torch = self.torch
        device = self.model.device
        loop = asyncio.get_event_loop()

        if cached_kv is None or cached_tokens <= 0:
            # Cold prefill: forward the full prompt.
            input_ids = torch.tensor(
                [list(request.prompt_tokens)], dtype=torch.long, device=device,
            )

            def _run_cold() -> Tuple[int, Any]:
                with torch.inference_mode():
                    out = self.model(input_ids=input_ids, use_cache=True)
                logits = out.logits[:, -1, :]
                tok = int(torch.argmax(logits, dim=-1).item())
                return tok, out.past_key_values

            first_token, past = await loop.run_in_executor(None, _run_cold)
        else:
            # Cache hit: clone the cached KV (so this request's growth doesn't
            # corrupt the shared cache) and forward only the suffix tokens.
            cached_past = cached_kv["past"]
            suffix = list(request.prompt_tokens[cached_tokens:])

            def _run_resume() -> Tuple[int, Any]:
                cloned = _clone_past(cached_past, device, torch)
                if not suffix:
                    # Prompt is exactly the cached prefix — re-run the last
                    # cached token so we can produce a next-token prediction.
                    truncated = _truncate_past(cloned, cached_tokens - 1, torch)
                    ids = torch.tensor(
                        [[request.prompt_tokens[cached_tokens - 1]]],
                        dtype=torch.long, device=device,
                    )
                    used_past = truncated
                else:
                    ids = torch.tensor([suffix], dtype=torch.long, device=device)
                    used_past = cloned
                with torch.inference_mode():
                    out = self.model(
                        input_ids=ids,
                        past_key_values=used_past,
                        use_cache=True,
                    )
                logits = out.logits[:, -1, :]
                tok = int(torch.argmax(logits, dim=-1).item())
                return tok, out.past_key_values

            first_token, past = await loop.run_in_executor(None, _run_resume)

        kv = {"past": past, "len": request.prompt_len, "next_pos": request.prompt_len}
        self._kv[request.request_id] = kv
        return first_token, kv

    async def snapshot_prefix(
        self,
        request_id: str,
        kv_state: Dict[str, Any],
        num_tokens: int,
    ) -> Dict[str, Any]:
        torch = self.torch
        loop = asyncio.get_event_loop()
        past = kv_state.get("past")
        if past is None:
            return {"past": None, "len": num_tokens, "next_pos": num_tokens}

        def _snapshot() -> Dict[str, Any]:
            return {
                "past": _truncate_past(past, num_tokens, torch),
                "len": num_tokens,
                "next_pos": num_tokens,
            }

        return await loop.run_in_executor(None, _snapshot)

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
        # In a real disaggregated deployment this is the NCCL/RDMA hop;
        # in-process we do an explicit ``.to(device)`` for each layer's K/V.
        # IMPORTANT: build a fresh dict (and a fresh DynamicCache) instead of
        # mutating ``kv_state`` in place — the same dict reference is held by
        # the prefill engine's still-active seq, and its scheduler may run a
        # spurious step() between handoff and eviction. Mutating would leave
        # the prefill seq pointing at a cache on the wrong GPU.
        torch = self.torch
        target = self.model.device
        past = kv_state.get("past")
        moved_past = _move_past_kv(past, target, torch) if past is not None else None
        new_kv = {**kv_state, "past": moved_past}
        self._kv[request_id] = new_kv
        return new_kv

    async def release(self, request_id: str) -> None:
        self._kv.pop(request_id, None)


def _move_past_kv(past, target_device, torch):
    """Return a NEW past_key_values on ``target_device``. Does not mutate the
    input — both DynamicCache and legacy-tuple inputs come back as fresh
    objects so the source engine's references aren't disturbed."""
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        from transformers import DynamicCache  # type: ignore
        out = DynamicCache()
        new_keys = []
        new_vals = []
        seen = 0
        for k, v in zip(past.key_cache, past.value_cache):
            kk = k.to(target_device) if k.device != target_device else k
            vv = v.to(target_device) if v.device != target_device else v
            new_keys.append(kk)
            new_vals.append(vv)
            seen = kk.shape[-2]
        out.key_cache = new_keys
        out.value_cache = new_vals
        out._seen_tokens = seen
        return out
    # Legacy tuple-of-tuples form.
    moved = []
    for k, v in past:
        kk = k.to(target_device) if k.device != target_device else k
        vv = v.to(target_device) if v.device != target_device else v
        moved.append((kk, vv))
    return tuple(moved)


def _clone_past(past, target_device, torch):
    """Deep-copy past_key_values onto ``target_device`` so writes by a later
    forward call don't mutate the shared prefix cache. Sets the destination
    DynamicCache's lists directly to avoid ``update()``'s placeholder-on-CPU
    behavior — see ``_truncate_past`` for the same pattern."""
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        from transformers import DynamicCache  # type: ignore
        out = DynamicCache()
        new_keys = []
        new_vals = []
        seen = 0
        for k, v in zip(past.key_cache, past.value_cache):
            kk = k.to(target_device, copy=True).detach()
            vv = v.to(target_device, copy=True).detach()
            new_keys.append(kk)
            new_vals.append(vv)
            seen = kk.shape[-2]
        out.key_cache = new_keys
        out.value_cache = new_vals
        out._seen_tokens = seen
        return out
    return tuple(
        (k.to(target_device, copy=True).detach(),
         v.to(target_device, copy=True).detach())
        for k, v in past
    )


def _truncate_past(past, num_tokens, torch):
    """Return a detached past_key_values containing only the first
    ``num_tokens`` of context per layer. Cheap to call — used both for
    snapshotting a prefix and for the prompt==prefix edge case.

    Builds the new DynamicCache by setting its key_cache/value_cache lists
    directly (without going through ``update()``). Going through ``update``
    appends ``torch.tensor([])`` placeholders for any "skipped" layers using
    the default device (CPU), which contaminates a later forward pass; direct
    assignment avoids that path entirely.
    """
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        from transformers import DynamicCache  # type: ignore
        out = DynamicCache()
        new_keys = []
        new_vals = []
        for k, v in zip(past.key_cache, past.value_cache):
            new_keys.append(k[:, :, :num_tokens, :].clone().detach())
            new_vals.append(v[:, :, :num_tokens, :].clone().detach())
        out.key_cache = new_keys
        out.value_cache = new_vals
        out._seen_tokens = num_tokens
        return out
    return tuple(
        (k[:, :, :num_tokens, :].clone().detach(),
         v[:, :, :num_tokens, :].clone().detach())
        for k, v in past
    )


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
    # On a pure-prefill engine in the disaggregated path the seq lives in
    # _active only briefly (until handoff + evict). Skip it in step() so the
    # prefill scheduler doesn't waste compute decoding a sequence that's
    # about to be evicted.
    decode_eligible: bool = True


@dataclass
class _PrefixEntry:
    """A cached prefix's state on this engine.

    The bookkeeping side: ``block_ids`` are pinned in the engine's KVAllocator
    under a synthetic owner id (``__prefix__:<hash>``) so a request that hits
    this prefix can re-use those block ids via refcount, and so the blocks
    survive the originating request's completion.

    The compute side: ``kv_snapshot`` is a backend-specific opaque handle —
    for ``TransformersBackend`` it's a cloned ``DynamicCache`` of length
    ``num_tokens``; for ``MockBackend`` it's metadata only.
    """

    prefix_hash: str
    num_tokens: int
    num_blocks: int
    block_ids: List[int]
    kv_snapshot: Any
    last_used: float = 0.0


def _prefix_owner_id(prefix_hash: str) -> str:
    return f"__prefix__:{prefix_hash}"


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
        prefix_cache_capacity: int = 32,
        prefix_cache_min_blocks: int = 1,
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

        # Prefix cache: hash -> _PrefixEntry. OrderedDict gives LRU semantics
        # (oldest at head, recently-used at tail).
        self._prefix_index: "OrderedDict[str, _PrefixEntry]" = OrderedDict()
        self._prefix_capacity = prefix_cache_capacity
        self._prefix_min_blocks = prefix_cache_min_blocks
        # Stats.
        self.prefix_hits: int = 0
        self.prefix_misses: int = 0
        self.prefix_tokens_saved: int = 0  # cumulative tokens skipped via cache

    # ---- prefix cache helpers --------------------------------------------------

    def _max_prefix_blocks_for_request(self, request: Request) -> int:
        block_size = self.partition.block_size_tokens
        target = request.prefix_len_for_cache // block_size
        # Leave at least one suffix token so the backend always has a forward
        # to run (and so duplicate-prompt requests still produce a next token).
        upper = max(0, (request.prompt_len - 1) // block_size)
        return min(target, upper)

    def _prefix_lookup(self, request: Request) -> Optional[_PrefixEntry]:
        h = request.prefix_hash()
        entry = self._prefix_index.get(h)
        if entry is None:
            return None
        # Need at least one suffix token to forward. (Same condition we used
        # when caching, but enforce on the lookup side too.)
        if request.prompt_len <= entry.num_tokens:
            return None
        # Touch for LRU.
        self._prefix_index.move_to_end(h)
        entry.last_used = time.time()
        return entry

    async def _maybe_cache_prefix(
        self, request: Request, alloc: KVAllocation, kv: Dict[str, Any]
    ) -> None:
        cache_blocks = self._max_prefix_blocks_for_request(request)
        if cache_blocks < self._prefix_min_blocks:
            return
        cache_tokens = cache_blocks * self.partition.block_size_tokens
        h = request.prefix_hash()
        if h in self._prefix_index:
            # Already cached (a duplicate raced in or this is a re-prefill).
            self._prefix_index.move_to_end(h)
            return

        snapshot = await self.backend.snapshot_prefix(
            request.request_id, kv, cache_tokens,
        )
        prefix_block_ids = list(alloc.block_ids[:cache_blocks])
        owner_id = _prefix_owner_id(h)
        try:
            self.allocator.allocate(
                request_id=owner_id,
                num_tokens=cache_tokens,
                preemptible=False,
                priority=0.0,
                share_block_ids=prefix_block_ids,
            )
        except (ValueError, RuntimeError):
            # Allocator already has this synthetic owner registered.
            return

        self._prefix_index[h] = _PrefixEntry(
            prefix_hash=h,
            num_tokens=cache_tokens,
            num_blocks=cache_blocks,
            block_ids=prefix_block_ids,
            kv_snapshot=snapshot,
            last_used=time.time(),
        )
        self._evict_prefix_if_full()

    def _evict_prefix_if_full(self) -> None:
        while len(self._prefix_index) > self._prefix_capacity:
            old_hash, _ = self._prefix_index.popitem(last=False)
            # Drops the synthetic owner's refcount; blocks return to free pool
            # iff no live request currently shares them.
            self.allocator.free(_prefix_owner_id(old_hash))

    def prefix_cache_snapshot(self) -> Dict[str, int]:
        """Map of prefix_hash -> cached blocks, for the router's prefix_cache hint."""
        return {h: e.num_blocks for h, e in self._prefix_index.items()}

    # ---- request lifecycle (called by scheduler) -------------------------------

    async def admit_prefill(self, request: Request) -> int:
        """Allocate KV for prompt, run prefill, return first token.

        On a prefix-cache hit, the allocator shares the cached prefix's blocks
        via refcount and the backend resumes prefill from the cached position
        — saving both KV memory and the prefix's prefill compute.
        """
        entry = self._prefix_lookup(request)
        if entry is not None:
            return await self._admit_prefill_with_cache(request, entry)
        return await self._admit_prefill_fresh(request)

    async def _admit_prefill_fresh(self, request: Request) -> int:
        alloc = self.allocator.allocate(
            request_id=request.request_id,
            num_tokens=request.prompt_len,
            preemptible=True,
            priority=request.priority,
        )
        self.prefix_misses += 1
        t0 = time.perf_counter()
        first_token, kv = await self.backend.prefill(request)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        self.recent_ttft_ms = ttft_ms

        seq = _ActiveSeq(
            request=request,
            alloc=alloc,
            last_token=first_token,
            kv_state=kv,
            decode_eligible=(self.role != EngineRole.PREFILL),
        )
        seq.output.append(first_token)
        seq.tokens_produced = 1
        self._active[request.request_id] = seq
        self._completion_events[request.request_id] = asyncio.Event()

        # Snapshot a prefix-portion of this request's KV for future hits.
        await self._maybe_cache_prefix(request, alloc, kv)
        return first_token

    async def _admit_prefill_with_cache(
        self, request: Request, entry: _PrefixEntry,
    ) -> int:
        # Refcount-share the cached prefix's blocks; allocate fresh blocks
        # only for the suffix.
        alloc = self.allocator.allocate(
            request_id=request.request_id,
            num_tokens=request.prompt_len,
            preemptible=True,
            priority=request.priority,
            share_block_ids=entry.block_ids,
        )
        self.prefix_hits += 1
        self.prefix_tokens_saved += entry.num_tokens
        t0 = time.perf_counter()
        first_token, kv = await self.backend.prefill(
            request, cached_kv=entry.kv_snapshot, cached_tokens=entry.num_tokens,
        )
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        self.recent_ttft_ms = ttft_ms

        seq = _ActiveSeq(
            request=request,
            alloc=alloc,
            last_token=first_token,
            kv_state=kv,
            decode_eligible=(self.role != EngineRole.PREFILL),
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
        """One iteration: advance every decode-eligible sequence by one token.
        Returns finished ids."""
        # Skip seqs that aren't ready to decode (e.g., prefill seqs in the
        # disaggregated path that are awaiting handoff + eviction).
        eligible = [(rid, seq) for rid, seq in list(self._active.items())
                    if seq.decode_eligible]
        if not eligible:
            return []
        finished: List[str] = []
        t0 = time.perf_counter()

        async def _one(rid: str, seq: _ActiveSeq) -> Optional[str]:
            tok = await self.backend.decode_step(rid, seq.last_token, seq.kv_state)
            seq.output.append(tok)
            seq.last_token = tok
            seq.tokens_produced += 1
            try:
                self.allocator.extend(rid, extra_tokens=1)
            except RuntimeError:
                seq.done.set()
                return rid

            if seq.tokens_produced >= seq.request.max_new_tokens:
                seq.done.set()
                return rid
            return None

        results = await asyncio.gather(*[_one(rid, seq) for rid, seq in eligible])
        for rid in results:
            if rid is not None:
                finished.append(rid)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n = max(1, len(eligible))
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
            "prefix_cache": {
                "entries": len(self._prefix_index),
                "capacity": self._prefix_capacity,
                "hits": self.prefix_hits,
                "misses": self.prefix_misses,
                "tokens_saved": self.prefix_tokens_saved,
                "pinned_blocks": sum(e.num_blocks for e in self._prefix_index.values()),
            },
        }
