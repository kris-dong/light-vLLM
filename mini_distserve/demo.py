"""
End-to-end demo for mini_distserve.

Default mode runs the real Qwen2.5-7B-Instruct-AWQ model locally:
  * Probes each visible GPU for free memory (no hardcoded budgets).
  * Picks two GPUs with enough headroom for weights + KV.
  * Subtracts the model's quantized weight footprint and an activation
    reserve from the free memory to derive the per-engine KV budget.
  * Spins up 1 prefill engine + 1 decode engine on those GPUs, both
    running the AWQ model via ``TransformersBackend``.
  * Routes prompts through the disaggregated path (prefill -> KV transfer
    -> decode) and prints the decoded output.

Prompts are formatted with the model's chat template (system + user
messages, same shape as ``vllm_export/query_llm.py`` would send to a
vLLM OpenAI server), so this is the local equivalent of that HTTP
query — except instead of calling out to a vLLM server, we run the
prefill / decode forwards inside this process via the framework.

CLI:
    --prompt "..."           a single prompt (repeatable)
    --prompts-file PATH      newline-separated prompts file
    --num-requests N         pick the first N built-in demo prompts (default)
    --system "..."           system message (default: concise helpful)
    --max-new N              max tokens to generate per prompt
    --mock                   skip GPU/model; orchestration smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import List, Tuple

from .engine import Engine, MockBackend
from .partition import ModelPartition, ParallelLayout
from .router import EngineRole, Request, RouterConfig
from .serving import ServingSystem


QWEN_AWQ_PATH_DEFAULT = (
    "/scratch/kris/local-llm/.hf-cache-vllm-export/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct-AWQ"
)

# How much VRAM to leave on each GPU for activations + framework overhead
# beyond weights and the KV cache pool itself. ~1.5 GiB is enough for prefill
# of a few-thousand-token prompt at fp16 on Qwen-7B.
ACTIVATION_RESERVE_BYTES = int(1.5 * (1024 ** 3))

# Minimum KV pool we'll accept per engine. If a GPU can't fit this much KV
# after weights + activations, we skip it.
MIN_KV_BUDGET_BYTES = int(1.0 * (1024 ** 3))


def make_request(rid: str, prompt_tokens: List[int], *, max_new: int = 64,
                 expected: int = 32, ttft_ms: float = 4000.0,
                 tpot_ms: float = 200.0, priority: float = 1.0,
                 model_name: str = "qwen2.5-7b-awq") -> Request:
    return Request(
        request_id=rid,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new,
        expected_new_tokens=expected,
        ttft_slo_ms=ttft_ms,
        tpot_slo_ms=tpot_ms,
        priority=priority,
    )


def _probe_gpu_free_memory() -> List[Tuple[int, int, int, str]]:
    """Returns ``[(device_id, free_bytes, total_bytes, name), ...]``.

    Uses ``torch.cuda.mem_get_info`` so the value reflects what's actually
    free at this moment — other processes' allocations are accounted for.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA devices visible to this process")
    out: List[Tuple[int, int, int, str]] = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        props = torch.cuda.get_device_properties(i)
        out.append((i, int(free), int(total), props.name))
    return out


def _pick_two_gpus(gpus: List[Tuple[int, int, int, str]],
                   weight_bytes: int) -> Tuple[int, int]:
    """Pick the two GPUs with the most free memory that can each hold
    ``weight_bytes + ACTIVATION_RESERVE + MIN_KV_BUDGET``."""
    needed = weight_bytes + ACTIVATION_RESERVE_BYTES + MIN_KV_BUDGET_BYTES
    eligible = sorted(
        [g for g in gpus if g[1] >= needed],
        key=lambda g: g[1],
        reverse=True,
    )
    if len(eligible) >= 2:
        return eligible[0][0], eligible[1][0]
    if len(eligible) == 1:
        # Fall back to colocated on the only roomy GPU.
        return eligible[0][0], eligible[0][0]
    raise RuntimeError(
        f"no GPU has enough free memory: need ~{needed/(1024**3):.1f} GiB, "
        f"max free is {max(g[1] for g in gpus)/(1024**3):.1f} GiB"
    )


def _kv_budget_for_gpu(free_bytes: int, weight_bytes: int) -> int:
    budget = free_bytes - weight_bytes - ACTIVATION_RESERVE_BYTES
    return max(MIN_KV_BUDGET_BYTES, budget)


DEFAULT_SYSTEM_MSG = "You are a concise helpful assistant."

DEFAULT_PROMPTS = [
    "Explain what KV cache is in transformers, in two sentences.",
    "Write a haiku about disaggregated LLM serving.",
    "Why does prefill have different compute characteristics than decode?",
    "List three reasons to separate prefill and decode pools.",
    "In one paragraph, what is paged attention?",
    "Describe TTFT vs TPOT in your own words.",
]


def _apply_chat_template(tok, system_msg: str, user_msg: str) -> List[int]:
    """Render (system, user) into a token list using the model's chat template
    — same payload shape as ``query_llm.py`` sends to vLLM's chat endpoint."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )


async def run_real_demo(
    *,
    num_requests: int,
    model_path: str,
    prompts: List[str],
    system_msg: str,
    max_new: int,
    sequential: bool = False,
) -> None:
    from transformers import AutoTokenizer
    from .engine import TransformersBackend
    import torch

    gpus = _probe_gpu_free_memory()
    print("[gpu] visible:")
    for dev, free, total, name in gpus:
        print(f"  cuda:{dev}  {name}  free={free/(1024**3):5.2f} GiB / "
              f"total={total/(1024**3):5.2f} GiB")
    free_before = {g[0]: g[1] for g in gpus}

    # Size the model once on a tentative layout so we know the weight footprint
    # before we pick GPUs. Both prefill and decode engines use the same TP=1/PP=1
    # shape so the weight estimate is identical.
    tentative_layout = ParallelLayout(
        tensor_parallel_size=1, pipeline_parallel_size=1, device_ids=(0,)
    )
    sized = ModelPartition.from_hf_config(
        model_path=model_path,
        layout=tentative_layout,
        # Placeholder; we recompute the budget per engine below.
        per_gpu_kv_budget_bytes=int(1 * (1024 ** 3)),
        block_size_tokens=16,
        model_name="qwen2.5-7b-awq",
    )
    weight_bytes = sized.weight_bytes_total
    print(f"[partition] estimated weight footprint = "
          f"{weight_bytes/(1024**3):.2f} GiB ({sized.quant_method}-"
          f"{sized.quant_bits}bit)")

    prefill_dev, decode_dev = _pick_two_gpus(gpus, weight_bytes)
    if prefill_dev == decode_dev:
        print(f"[gpu] only one eligible GPU (cuda:{prefill_dev}); "
              f"running colocated on it.")
    else:
        print(f"[gpu] prefill on cuda:{prefill_dev}, decode on cuda:{decode_dev}")

    free_by_dev = {g[0]: g[1] for g in gpus}
    p_layout = ParallelLayout(
        tensor_parallel_size=1, pipeline_parallel_size=1,
        device_ids=(prefill_dev,),
    )
    d_layout = ParallelLayout(
        tensor_parallel_size=1, pipeline_parallel_size=1,
        device_ids=(decode_dev,),
    )

    # Real KV budget per engine: free - weights - activations.
    p_budget = _kv_budget_for_gpu(free_by_dev[prefill_dev], weight_bytes)
    d_budget = _kv_budget_for_gpu(free_by_dev[decode_dev], weight_bytes)
    if prefill_dev == decode_dev:
        # Two engines on the same GPU share its free memory; split it.
        shared = _kv_budget_for_gpu(free_by_dev[prefill_dev], 2 * weight_bytes)
        p_budget = d_budget = shared // 2

    print(f"[partition] KV budget: prefill={p_budget/(1024**3):.2f} GiB  "
          f"decode={d_budget/(1024**3):.2f} GiB")

    p_partition = ModelPartition.from_hf_config(
        model_path=model_path,
        layout=p_layout,
        per_gpu_kv_budget_bytes=p_budget,
        block_size_tokens=16,
        model_name="qwen2.5-7b-awq",
    )
    d_partition = ModelPartition.from_hf_config(
        model_path=model_path,
        layout=d_layout,
        per_gpu_kv_budget_bytes=d_budget,
        block_size_tokens=16,
        model_name="qwen2.5-7b-awq",
    )
    print(p_partition.summary())
    print(d_partition.summary())

    print("[load] instantiating prefill engine...")
    prefill_engine = Engine(
        engine_id=f"prefill-{prefill_dev}",
        role=EngineRole.PREFILL,
        partition=p_partition,
        backend=TransformersBackend(p_partition),
        max_batch_size=4,
    )
    print("[load] instantiating decode engine...")
    decode_engine = Engine(
        engine_id=f"decode-{decode_dev}",
        role=EngineRole.DECODE,
        partition=d_partition,
        backend=TransformersBackend(d_partition),
        max_batch_size=8,
    )

    sys = ServingSystem(
        RouterConfig(kv_block_tokens=16, min_free_kv_blocks_after_admit=2)
    )
    sys.register_engine(prefill_engine, max_batch_size=4)
    sys.register_engine(decode_engine, max_batch_size=8)
    await sys.start(report_period_s=0.1)

    # Snapshot GPU memory after model load — proves weights actually landed.
    print("[gpu] after model load:")
    for dev in sorted({prefill_dev, decode_dev}):
        free_now, total = torch.cuda.mem_get_info(dev)
        delta = (free_before[dev] - free_now) / (1024 ** 3)
        print(f"  cuda:{dev}  free={free_now/(1024**3):5.2f} GiB  "
              f"(delta from start: -{delta:5.2f} GiB)")

    # Confirm the loaded modules are quantized AWQ linears.
    qmod = type(next(prefill_engine.backend.model.parameters()).data).__name__
    awq_linear_ct = sum(
        1 for m in prefill_engine.backend.model.modules()
        if "WQLinear" in type(m).__name__ or "AwqLinear" in type(m).__name__
    )
    print(f"[model] prefill model class={type(prefill_engine.backend.model).__name__}  "
          f"awq_linear_modules={awq_linear_ct}  param_dtype={qmod}")

    # Tokenize prompts via the model's chat template — same payload shape as
    # vllm_export/query_llm.py's chat-completion request.
    tok = AutoTokenizer.from_pretrained(p_partition.model_path)
    prompts = (prompts or DEFAULT_PROMPTS)[: max(1, num_requests)]

    requests = []
    for i, p in enumerate(prompts):
        token_ids = _apply_chat_template(tok, system_msg, p)
        requests.append(
            make_request(
                f"req-{i:03d}",
                token_ids,
                max_new=max_new,
                expected=max(1, max_new // 2),
                model_name=p_partition.model_name,
            )
        )

    async def run_one(idx: int, prompt_text: str, req: Request) -> None:
        t0 = time.perf_counter()
        try:
            r = await sys.submit(req, disaggregated=(prefill_dev != decode_dev))
        except Exception as e:
            print(f"[err]  {req.request_id}: {e!r}")
            return
        elapsed = (time.perf_counter() - t0) * 1000.0
        # The first token in r.output_tokens is from prefill; the rest from
        # decode. Drop any chat-template control tokens for display.
        text = tok.decode(r.output_tokens, skip_special_tokens=True).strip()
        # Per-engine prefix-cache stats so the user sees the cache fill up.
        psnap = prefill_engine.snapshot()["prefix_cache"]
        print("─" * 72)
        print(f"[req {idx}] prompt: {prompt_text!r}")
        print(f"[req {idx}] tokens_in={req.prompt_len}  "
              f"tokens_out={len(r.output_tokens)}  "
              f"ttft={r.ttft_ms:6.1f}ms  total={elapsed:6.1f}ms  "
              f"prefill={r.prefill_engine}  decode={r.decode_engine}")
        print(f"[req {idx}] prefix_cache(prefill): hits={psnap['hits']} "
              f"misses={psnap['misses']} entries={psnap['entries']} "
              f"tokens_saved={psnap['tokens_saved']}")
        print(f"[req {idx}] response: {text}")

    if sequential:
        for i, (p, r) in enumerate(zip(prompts, requests)):
            await run_one(i, p, r)
    else:
        await asyncio.gather(*[
            run_one(i, p, r) for i, (p, r) in enumerate(zip(prompts, requests))
        ])

    print("─" * 72)
    print(f"[final] prefill={prefill_engine.snapshot()}")
    print(f"[final] decode= {decode_engine.snapshot()}")
    await sys.stop()


async def run_mock_demo(num_requests: int) -> None:
    layout_p = ParallelLayout(tensor_parallel_size=1, pipeline_parallel_size=1, device_ids=(0,))
    layout_d = ParallelLayout(tensor_parallel_size=1, pipeline_parallel_size=1, device_ids=(1,))

    fake_partition = ModelPartition(
        model_name="qwen2.5-7b-awq",
        model_path="(mock)",
        layout=layout_p,
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        hidden_size=3584,
        max_position_embeddings=32768,
        dtype="fp16",
        kv_bytes_per_token_per_layer=2 * 4 * 128 * 2,
        kv_bytes_per_token=28 * 2 * 4 * 128 * 2,
        block_size_tokens=16,
        kv_block_bytes=16 * 28 * 2 * 4 * 128 * 2,
        total_kv_blocks=2048,
    )
    decode_partition = ModelPartition(**{**fake_partition.__dict__, "layout": layout_d})

    prefill_engine = Engine(
        engine_id="prefill-0",
        role=EngineRole.PREFILL,
        partition=fake_partition,
        backend=MockBackend(fake_partition),
        max_batch_size=64,
    )
    decode_engine = Engine(
        engine_id="decode-0",
        role=EngineRole.DECODE,
        partition=decode_partition,
        backend=MockBackend(decode_partition),
        max_batch_size=128,
    )

    sys = ServingSystem(RouterConfig(kv_block_tokens=16, min_free_kv_blocks_after_admit=8))
    sys.register_engine(prefill_engine)
    sys.register_engine(decode_engine)
    await sys.start(report_period_s=0.05)

    print(f"[setup] prefill={prefill_engine.snapshot()}")
    print(f"[setup] decode= {decode_engine.snapshot()}")

    async def run_one(i: int) -> None:
        prompt = [100 + (i * 17 + j) % 1000 for j in range(200 + i * 50)]
        req = make_request(f"req-{i:03d}", prompt, max_new=24 + i * 4, expected=16 + i * 2)
        t0 = time.perf_counter()
        result = await sys.submit(req, disaggregated=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(
            f"[done] {req.request_id}  prompt_len={req.prompt_len:4d}  "
            f"out={len(result.output_tokens)}  ttft={result.ttft_ms:6.1f}ms  "
            f"total={elapsed:6.1f}ms  prefill={result.prefill_engine}  "
            f"decode={result.decode_engine}  spilled={result.spilled}"
        )

    await asyncio.gather(*[run_one(i) for i in range(num_requests)])

    print(f"[final] prefill={prefill_engine.snapshot()}")
    print(f"[final] decode= {decode_engine.snapshot()}")
    await sys.stop()


def _load_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend([ln.strip() for ln in f if ln.strip()])
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="Skip GPU + real model load; run MockBackend.")
    p.add_argument("--model-path", default=QWEN_AWQ_PATH_DEFAULT,
                   help="HF cache dir or snapshot dir for the AWQ model.")
    p.add_argument("--prompt", action="append", default=None,
                   help="A prompt to send. Repeatable.")
    p.add_argument("--prompts-file", default=None,
                   help="Newline-separated prompts file.")
    p.add_argument("--system", default=DEFAULT_SYSTEM_MSG,
                   help="System message (default: concise helpful assistant).")
    p.add_argument("--max-new", type=int, default=128,
                   help="Max new tokens to generate per prompt.")
    p.add_argument("--num-requests", type=int, default=2,
                   help="If no --prompt/--prompts-file given, use the first N "
                        "built-in demo prompts.")
    p.add_argument("--sequential", action="store_true",
                   help="Submit prompts one at a time. Reveals prefix-cache "
                        "hits when prompts share a prefix; otherwise "
                        "concurrent submits all start before any caches.")
    args = p.parse_args()

    if args.mock:
        asyncio.run(run_mock_demo(args.num_requests))
        return

    user_prompts = _load_prompts(args)
    asyncio.run(run_real_demo(
        num_requests=args.num_requests if not user_prompts else len(user_prompts),
        model_path=args.model_path,
        prompts=user_prompts,
        system_msg=args.system,
        max_new=args.max_new,
        sequential=args.sequential,
    ))


if __name__ == "__main__":
    main()
