# `mini_distserve` — A miniature disaggregated LLM serving framework

The package implements a small-scale model of a disaggregated LLM serving system (DistServe / Splitwise-style) where prefill and decode run on separate engine pools. Below is a per-file breakdown of every function.

```bash
 ┌──────────────────────────────┬─────────────────┬────────────────────────────────────────┐    
  │            Signal            │      Value      │                Meaning                 │    
  ├──────────────────────────────┼─────────────────┼────────────────────────────────────────┤ 
  │ awq_linear_modules=196       │ 28 layers × 7   │ Real AWQ-quantized linear modules in   │    
  │                              │ linears         │ the loaded model                       │  
  ├──────────────────────────────┼─────────────────┼────────────────────────────────────────┤    
  │ model class=Qwen2ForCausalLM │ —               │ Actual transformers Qwen2ForCausalLM,  │
  │                              │                 │ not a stub                             │    
  ├──────────────────────────────┼─────────────────┼────────────────────────────────────────┤  
  │ cuda:2 delta: -4.71 GiB      │ After load      │ Quantized weights physically landed on │    
  │                              │                 │  GPU2                                  │    
  ├──────────────────────────────┼─────────────────┼────────────────────────────────────────┤
  │ "The capital of France is    │ —               │ Real argmax output from the AWQ        │    
  │ Paris."                      │                 │ Qwen-7B forward pass                   │    
  └──────────────────────────────┴─────────────────┴────────────────────────────────────────┘
                                                                                                 
  The mini_distserve.demo now mirrors query_llm.py's payload shape:                              
  
  ┌────────────────────────────────────┬─────────────────────────────────────────────────────┐   
  │     query_llm.py (HTTP → vLLM)     │          mini_distserve.demo (in-process)           │ 
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤   
  │ Builds [{system}, {user}] chat     │ Same — _apply_chat_template()                       │ 
  │ messages                           │                                                     │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤   
  │ POST /v1/chat/completions to vLLM  │ ServingSystem.submit() to local engines             │   
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤   
  │ vLLM does prefill+decode           │ Our prefill engine (cuda:2) → KV transfer → decode  │   
  │ internally                         │ engine (cuda:0)                                     │ 
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤   
  │ Builds [{system}, {user}] chat     │ Same — _apply_chat_template()                       │
  │ messages                           │                                                     │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ POST /v1/chat/completions to vLLM  │ ServingSystem.submit() to local engines             │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ vLLM does prefill+decode           │ Our prefill engine (cuda:2) → KV transfer → decode  │
  │ internally                         │ engine (cuda:0)                                     │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Returns choices[0].message.content │ Tokenizer.decode of r.output_tokens                 │
  └────────────────────────────────────┴─────────────────────────────────────────────────────
```
## Downloading the model from HuggingFace

The demo defaults to **`Qwen/Qwen2.5-7B-Instruct-AWQ`** (~5 GiB of AWQ INT4 weights). It's a public model — no HF login required — but you do need to materialize a snapshot on disk before the demo can load it. The recommended cache root for this repo is `/scratch/kris/local-llm/.hf-cache-vllm-export` so the same weights are reused by both `mini_distserve` and the `vllm_export` Docker setup.

### 1) Install the HuggingFace client

```bash
/scratch/kris/local-llm/.venv/bin/pip install -U "huggingface_hub[cli]"
```

### 2) Point HF at the shared cache

```bash
export HF_HOME=/scratch/kris/local-llm/.hf-cache-vllm-export
# (Optional) Only needed for gated repos; Qwen2.5-7B-Instruct-AWQ is public.
# export HF_TOKEN=hf_xxx
```

### 3) Download the snapshot

Either via the CLI:

```bash
/scratch/kris/local-llm/.venv/bin/huggingface-cli download \
    Qwen/Qwen2.5-7B-Instruct-AWQ \
    --local-dir-use-symlinks False
```

Or from Python (equivalent — pulls the same files into the same cache):

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
    cache_dir="/scratch/kris/local-llm/.hf-cache-vllm-export",
)
```

### 4) Verify the layout

After download, the cache should look like:

```
/scratch/kris/local-llm/.hf-cache-vllm-export/
└── hub/
    └── models--Qwen--Qwen2.5-7B-Instruct-AWQ/
        ├── blobs/
        ├── refs/
        └── snapshots/<sha>/
            ├── config.json
            ├── model-00001-of-00002.safetensors
            ├── model-00002-of-00002.safetensors
            ├── tokenizer.json
            └── ...
```

`partition.py::_resolve_snapshot_dir` accepts either form: the `models--ORG--NAME/` cache directory directly, or a specific `snapshots/<sha>/` snapshot directory. The demo's `--model-path` flag takes either.

### 5) (Optional) Pre-flight test

Confirm the snapshot loads before running the full demo:

```bash
/scratch/kris/local-llm/.venv/bin/python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
    '/scratch/kris/local-llm/.hf-cache-vllm-export/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ',
    trust_remote_code=False,
)
print(cfg.model_type, cfg.num_hidden_layers, getattr(cfg, 'quantization_config', None))
"
```

You should see `qwen2 28 {'quant_method': 'awq', 'bits': 4, ...}`.

---

## How to run

The default path runs the real `Qwen2.5-7B-Instruct-AWQ` model end-to-end on the local GPUs.

```bash
# Real model (default). Probes each GPU's free memory at runtime,
# sizes the KV pool to (free − weights − 1.5 GiB activation reserve),
# and routes prompts through prefill -> KV transfer -> decode.
/scratch/kris/local-llm/.venv/bin/python -m mini_distserve.demo --num-requests 4

# Custom model path (must be either a snapshot dir with config.json or
# an HF cache dir of the form models--ORG--NAME/).
/scratch/kris/local-llm/.venv/bin/python -m mini_distserve.demo \
    --model-path /scratch/kris/local-llm/.hf-cache-vllm-export/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ \
    --num-requests 4

# Orchestration-only smoke test (no GPU, no model load).
/scratch/kris/local-llm/.venv/bin/python -m mini_distserve.demo --mock --num-requests 4
```

Run from `/scratch/kris/local-llm` so the package import resolves. Dependencies that must be present in the venv: `torch`, `transformers==4.51.3`, `autoawq==0.2.9`, `accelerate` (the AWQ quantizer needs all of these; newer transformers switched the AWQ backend to `gptqmodel`, which doesn't build cleanly here).

---

## `__init__.py`
Public re-exports. Pulls `Engine`, `MockBackend`, `TransformersBackend`, `LLMRouter`, `Request`, `RouterConfig`, `KVAllocator`, `ModelPartition`, `ParallelLayout`, `LocalScheduler`, `ServingSystem` to the package root.

---

## `router.py` — Routing decisions across engines

**Types**
- `EngineRole` (enum) — `COLOCATED` / `PREFILL` / `DECODE`.
- `Request` (frozen dataclass) — request payload (id, model, tokens, SLOs, priority, prefix-cache window).
  - `prompt_len` — `len(prompt_tokens)`.
  - `predicted_new_tokens` — uses `expected_new_tokens` if given, else `0.5 * max_new_tokens`.
  - `total_predicted_tokens` — prompt + predicted.
  - `prefix_hash()` — SHA-256 of the first `prefix_len_for_cache` tokens for prefix-cache lookup.
- `EngineState` — snapshot of an engine the router uses to score routing (queue len, batch size, free/evictable KV, prefix cache, recent TTFT/TPOT, parallelism).
  - `supports(request, role)` — is this engine eligible? (matching model + role compatible).
- `RouterConfig` — KV block size, safety margin, cost-model coefficients, scoring weights.
- `RouteDecision` — `(request_id, engine_id, role, score, reason-dict)`.

**`LLMRouter`**
- `__init__(config)` — stores config + `engines: dict[id → EngineState]`.
- `update_engine_state(state)` — refresh the cached state for one engine, stamps `last_report_time_s`.
- `_blocks_for_tokens(n)` — ceil(n / kv_block_tokens).
- `_initial_kv_blocks_needed(req)` — blocks needed to hold the prompt KV.
- `_predicted_total_kv_blocks_needed(req)` — blocks for prompt + predicted (or max) decode.
- `_prefix_hit_blocks(engine, req)` — looks up `engine.prefix_cache[req.prefix_hash()]`.
- `_has_kv_capacity_for_admission(engine, req, role)` — feasibility test (free + evictable − needed ≥ safety margin).
- `_estimate_prefill_ms(req, engine)` — prompt-len × cost, with TP discount and a small TP comm penalty.
- `_estimate_decode_pressure_ms(req, engine)` — predicted_steps × per-step × batch-load × log-context.
- `_estimate_queue_delay_ms(engine)` — queue length × recent TPOT.
- `_kv_pressure_score(engine, req)` — used-ratio after admission; ∞ if it wouldn't fit.
- `_prefix_benefit_score(engine, req)` — fraction of the prompt's KV already cached.
- `_slo_risk_score(...)` — TTFT/TPOT exceedance vs SLO, scaled by priority.
- `_batch_pressure_score(engine)` — current_batch / max_batch.
- `_staleness_score(engine)` — penalty if `last_report_time_s` is old.
- `_score_engine(engine, req, role)` — feasibility + weighted sum of the above. Lower is better. Returns `(score, reasons)` or `None`.
- `route(request, role)` — score every engine and pick the lowest. Raises if no candidate.
- `route_disaggregated(request)` — calls `route` twice (PREFILL, then DECODE) and returns both decisions.

### What `risk` means in `_slo_risk_score`

`risk` is one component of the routing score the `LLMRouter` computes for each engine. Lower total score wins, and `risk` is a positive contribution (weighted by `w_slo_risk`, default 3.0), so **a higher `risk` makes the router less likely to send the request to that engine.**

Concretely, `risk` answers: *"How much does this engine's expected behavior overshoot the request's SLO?"*

It's measured as a **fractional overshoot**, not an absolute time:

| Predicted vs. SLO | Risk contribution |
|---|---|
| Predicted ≤ SLO (on-spec) | `0` |
| Predicted = 1.5× SLO | `0.5` |
| Predicted = 3× SLO | `2.0` |

The formula: `risk += max(0, predicted / SLO − 1)`.

Two SLOs feed in:
- **TTFT (time to first token)** — for engines that will run prefill. Compares `max(queue_delay + prefill_time, recent_ttft)` against the request's `ttft_slo_ms`. The `max(...)` means: trust the cost model's prediction normally, but never claim the engine will be faster than what it just demonstrated. Cold engines (`recent_ttft = 0`) fall through to the model-only path.
- **TPOT (time per output token)** — for engines that will run decode. Compares the engine's `recent_tpot_ms` directly against `tpot_slo_ms` (skipped while `recent_tpot_ms == 0`, i.e. before the engine has done any decode).

Both are summed, then divided by `request.priority` so high-priority requests get *less* effective penalty for the same predicted overshoot.

**Why fractions, not absolute milliseconds?** A 200 ms overshoot on a 100 ms SLO (latency-critical chat) is much worse than a 200 ms overshoot on a 5000 ms SLO (long-form generation). Normalizing by the SLO makes the score comparable across requests with different deadlines.

The `__main__` block is a self-test that registers four engines and prints colocated and disaggregated decisions.

---

## `kv_allocator.py` — Paged KV block allocator (vLLM-style, distilled)

**`KVAllocation`** — per-request handle: `request_id`, `block_ids`, `preemptible`, `priority`.

**`KVAllocator`**
- `__init__(total_blocks, block_size_tokens)` — initialize pool of free block ids, `_owners` dict, `_owner_order` (insertion order for eviction tiebreak), `_refcount` (for prefix sharing), and a `threading.Lock`.
- `blocks_for_tokens(n)` — ceil(n / block_size).
- `free_blocks` (property), `used_blocks` (property), `evictable_blocks()` — capacity introspection.
- `can_admit(num_tokens, safety_margin_blocks)` — admission feasibility.
- `allocate(request_id, num_tokens, preemptible, priority, share_block_ids)` — bumps refcount on shared blocks (prefix-cache hits), pops fresh block ids for the rest, registers ownership. Errors on duplicate id or out-of-KV.
- `extend(request_id, extra_tokens)` — append blocks to an existing allocation as decode grows it; returns the new block ids. Raises if pool exhausted.
- `free(request_id)` — drop refcount on each owned block; returns blocks actually released to the free set.
- `select_spill_victims(blocks_needed, protect_request_ids)` — pick preemptible owners (low-priority first, then oldest) until enough blocks would be reclaimed. Read-only — does not free.
- `_owner_order_index(rid)` — positional rank within `_owner_order` for tiebreaking.
- `snapshot()` — dict of (total/free/used/owners/evictable).
- `get_block_ids(rid)` — copy of an allocation's block list.

---

## `partition.py` — Model partitioning descriptor

Reads HF `config.json` and converts a TP/PP layout + per-GPU KV budget into a sized partition. Now also captures quantization metadata and an estimated weight footprint so the demo can size KV from *real* free memory instead of a hardcoded budget.

**Module-level**
- `_DTYPE_BYTES` — name → bytes/element (fp32=4, bf16/fp16=2, fp8/int8=1).
- `_resolve_snapshot_dir(model_path)` — accepts either a snapshot dir with `config.json` or an HF cache dir of the form `models--ORG--NAME/`; in the latter it walks `snapshots/<sha>/` to find the one with `config.json`.
- `_estimate_weight_bytes(...)` — rough per-engine weight footprint. Counts attention QKV+O and MLP up/gate/down per layer; for AWQ/GPTQ uses `bits/8` packing plus per-group fp16 scale+zero overhead; adds embedding, lm_head, and RMSNorm bytes.

**`ParallelLayout`** — `tensor_parallel_size`, `pipeline_parallel_size`, `device_ids`.
- `world_size` — TP × PP.
- `__post_init__` — validates `len(device_ids) == world_size`.

**`ModelPartition`** — sizing for one engine instance. New fields: `quant_method`, `quant_bits`, `weight_bytes_total`.
- `from_hf_config(model_path, layout, per_gpu_kv_budget_bytes, block_size_tokens, dtype_override, model_name)` — reads layers / kv-heads / head-dim / dtype from `config.json`, applies TP head-sharding and PP layer-sharding, computes `kv_bytes_per_token`, `kv_block_bytes`, and `total_kv_blocks` for the engine. KV is sized from the compute dtype (e.g. fp16) even when weights are AWQ — the K/V tensors are not quantized at runtime. Also reads `quantization_config` and calls `_estimate_weight_bytes`.
- `summary()` — multi-line printable summary, now including the quant scheme and weight footprint.

---

## `engine.py` — Prefill / decode workers

**Helper**
- `_resolve_loadable_id_and_cache(model_path)` — if local snapshot has `*.safetensors`/`*.bin`, load directly; otherwise reverse-engineer the HF repo id from `models--ORG--NAME/snapshots/...` and return cache root.
- `_move_past_kv(past, target_device, torch)` — moves HF `past_key_values` to a target device. Handles both the modern `DynamicCache` (mutates `key_cache`/`value_cache` lists in place) and the legacy tuple-of-tuples form. Used by the disaggregated KV handoff so prefill-side tensors get copied to the decode engine's GPU.

**`Backend` (Protocol)** — interface: `prefill`, `decode_step`, `receive_kv`, `release`.

**`MockBackend`** — simulated forward for orchestration tests.
- `__init__` — store partition + per-token timing constants and a `_kv` dict.
- `prefill(req)` — sleeps proportional to prompt length, returns a deterministic pseudo-token + KV-metadata.
- `decode_step(rid, prev_token, kv_state)` — sleeps one decode-step, returns a deterministic next token.
- `receive_kv(rid, kv_state)` — sleeps proportional to block count to emulate KV transfer.
- `release(rid)` — drops `_kv[rid]`.

**`TransformersBackend`** — real Qwen-7B / Qwen-7B-AWQ forward via HuggingFace `transformers`.
- `__init__(partition, max_new_tokens_cap)` — picks dtype, resolves load id, loads tokenizer + model. **AWQ/GPTQ models are pinned via `device_map={"": "cuda:N"}`** (their packed quantized linears don't survive `.to()` after load); dense models use plain `from_pretrained(...).to(device)`. Multi-GPU per engine still goes through `device_map="auto"` and requires `accelerate`.
- `prefill(req)` — runs prompt forward via `model(...)` in an executor, captures `past_key_values`, returns `(argmax_token, kv)`.
- `decode_step(rid, prev_token, kv_state)` — single-token forward with `past_key_values`, returns next token.
- `receive_kv(rid, kv_state)` — calls `_move_past_kv` to transfer the prefill engine's `past_key_values` onto this engine's GPU, then stashes it. This is the explicit cross-GPU hop that stands in for NCCL/RDMA in a real disaggregated deployment.
- `release(rid)` — drops cached KV.

**`_ActiveSeq`** — internal record per in-flight request: `request, alloc, last_token, kv_state, tokens_produced, output, done`.

**`Engine`**
- `__init__(engine_id, role, partition, backend, max_batch_size, kv_safety_margin_blocks)` — owns a `KVAllocator`, active/completed maps, and rolling TTFT/TPOT stats.
- `admit_prefill(req)` — allocate prompt-sized KV, call `backend.prefill`, record TTFT, register active seq, return first token.
- `admit_decode(req, first_token, kv_state)` — `backend.receive_kv`, allocate KV for `prompt + 1 + predicted_new`, register active seq.
- `step()` — one iteration: in parallel, advance every active seq by one token via `backend.decode_step`, extend KV by one token, mark sequences done if they hit `max_new_tokens` or run out of KV; updates `recent_tpot_ms`. Returns finished ids.
- `reap(rid)` — release backend state, free KV, stash `output` in `_completed`, set the completion event.
- `wait_for(rid)` — async wait on the completion event, then consume the stashed output.
- `evict(rid)` — spill: drop active seq, release backend state, free KV, wake any waiter.
- `export_kv(rid)` — return the `kv_state` of an active seq (used at prefill→decode handoff).
- `active_count()` / `snapshot()` — introspection.

---

## `scheduler.py` — Per-engine local scheduler

`AdmitOutcome` enum: `ADMITTED`, `QUEUED`, `SPILLED_AND_ADMITTED`, `REJECTED`. `AdmitResult` carries outcome + spilled victims.

**`LocalScheduler`**
- `__init__(engine, router_config)` — owns an `asyncio.Queue` for waiting requests, an `asyncio.Lock`, and a stop event.
- `_blocks_needed(req)` — blocks for prompt + predicted (or max) decode.
- `_slo_admissible(req)` — reject if engine's recent TTFT/TPOT already exceeds 1.5× the request SLO.
- `admit(req)` — main admission decision (see flow comment in the file): admit if free − need ≥ margin; otherwise pick spill victims to free `need + margin − free`; if none exist, queue (return a future).
- `run(step_period_s)` — iteration loop: while not stopped, `engine.step()` → reap finished → `_drain_wait_queue()`. Idles briefly when there is no work.
- `_drain_wait_queue()` — re-checks queued requests against currently free KV; admitted ones get their futures resolved, others go back into the queue.
- `stop()` / `queue_len()` — control + introspection.

---

## `serving.py` — Top-level wiring

**`ServeResult`** — output payload: tokens, prefill/decode engine ids, TTFT, total ms, spilled victims.

**`ServingSystem`**
- `__init__(router_config)` — owns the `LLMRouter`, engine + scheduler registries, task handles.
- `register_engine(engine, max_batch_size)` — adds engine + creates its `LocalScheduler`, pushes initial `EngineState` to the router.
- `_snapshot_state(engine, max_batch_size)` — build an `EngineState` from the engine's current allocator + scheduler stats.
- `start(report_period_s)` — launches one `sched.run()` task per engine and a `_reporter_loop` that periodically refreshes router state.
- `stop()` — sets stop flags, cancels tasks, awaits cancellation.
- `_reporter_loop(period_s)` — in a loop, push fresh `EngineState` snapshots into the router so routing is based on current load.
- `submit(request, disaggregated)` — main entry point.
  - Colocated: route → admit → `admit_prefill` → wait for completion.
  - Disaggregated: `route_disaggregated` → admit on prefill → `admit_prefill` (TTFT measured here) → `export_kv` → admit on decode → `admit_decode` (KV transfer + reservation) → evict prefill copy → wait for decode completion.
- `_admit_until_ok(sched, req)` — retry loop that handles `QUEUED` (sleep + retry), surfaces `REJECTED` as a `RuntimeError`.
- `_await_completion(engine, rid)` — `await engine.wait_for(rid)`.

---

## `demo.py` — End-to-end demo

Default mode is **real**: loads Qwen2.5-7B-Instruct-AWQ on real GPUs and routes prompts through the framework. `--mock` falls back to `MockBackend` for orchestration-only testing.

**Module-level constants**
- `QWEN_AWQ_PATH_DEFAULT` — local path to the AWQ model snapshot.
- `ACTIVATION_RESERVE_BYTES` — VRAM to leave free for runtime activations beyond weights + KV (~1.5 GiB).
- `MIN_KV_BUDGET_BYTES` — minimum KV pool we'll accept per engine (1 GiB); GPUs that can't fit this much after weights are skipped.

**Functions**
- `make_request(rid, prompt_tokens, ...)` — convenience constructor for a `Request`.
- `_probe_gpu_free_memory()` — calls `torch.cuda.mem_get_info` on every visible GPU; returns `[(device_id, free_bytes, total_bytes, name), ...]`. This is the real free memory at this instant, not a guess.
- `_pick_two_gpus(gpus, weight_bytes)` — picks the two GPUs with the most free memory that can each hold `weights + activation_reserve + min_kv_budget`. Falls back to colocated on one GPU if only one is eligible.
- `_kv_budget_for_gpu(free_bytes, weight_bytes)` — `free − weights − activation_reserve`, floored at `MIN_KV_BUDGET_BYTES`.
- `run_real_demo(num_requests, model_path)` — probes GPUs → estimates AWQ weight footprint via `ModelPartition.from_hf_config` → picks prefill/decode GPUs → builds two `ModelPartition`s with the per-GPU KV budgets derived from real free memory → instantiates two `Engine`s with `TransformersBackend` → registers them with `ServingSystem` → tokenizes a few real prompts → submits via the disaggregated path (or colocated if both engines landed on the same GPU) → prints decoded outputs and final KV snapshots.
- `run_mock_demo(num_requests)` — synthetic Qwen-shaped `ModelPartition` with `MockBackend`, no GPU work; useful as an orchestration smoke test.
- `main()` — argparse entry point: `--mock`, `--model-path`, `--num-requests`. Default dispatches to `run_real_demo`.

---

**End-to-end flow (disaggregated path):**
`submit` → `LLMRouter.route_disaggregated` (picks prefill + decode engine via weighted scoring) → `LocalScheduler.admit` on prefill (KV check / spill / queue) → `Engine.admit_prefill` (allocate prompt KV, run forward, get first token) → `Engine.export_kv` → `LocalScheduler.admit` on decode → `Engine.admit_decode` (`receive_kv` moves `past_key_values` cross-GPU + reserve decode KV) → `Engine.evict` on prefill side → iteration loop in `LocalScheduler.run` advances tokens via `Engine.step` → `wait_for` returns the completed token list.
