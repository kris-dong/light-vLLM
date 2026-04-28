"""
Serving system — top-level wiring.

Responsibilities:
  * Hold the LLMRouter and a registry of (engine, scheduler) pairs.
  * Periodically push EngineState snapshots to the router so routing decisions
    use fresh data.
  * Expose ``submit(request)`` which, in disaggregated mode:
        1. asks the router for a prefill engine,
        2. asks its scheduler to admit,
        3. runs prefill on that engine, gets first token + KV state,
        4. asks the router for a decode engine,
        5. asks its scheduler to admit,
        6. transfers KV (kv_handoff) and starts decode on that engine,
        7. waits for completion and returns the full token list.
    In colocated mode, steps 4-6 are skipped and decode happens on the same
    engine.

This is the file that the rest of the package wires through; everything below
it is just orchestration.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .engine import Engine
from .router import (
    EngineRole,
    EngineState,
    LLMRouter,
    Request,
    RouteDecision,
    RouterConfig,
)
from .scheduler import AdmitOutcome, AdmitResult, LocalScheduler


@dataclass
class ServeResult:
    request_id: str
    output_tokens: List[int]
    prefill_engine: str
    decode_engine: str
    ttft_ms: float
    total_ms: float
    spilled: List[str]


class ServingSystem:
    def __init__(self, router_config: Optional[RouterConfig] = None) -> None:
        self.router = LLMRouter(router_config or RouterConfig())
        self.engines: Dict[str, Engine] = {}
        self.schedulers: Dict[str, LocalScheduler] = {}
        self._engine_tasks: Dict[str, asyncio.Task] = {}
        self._reporter_task: Optional[asyncio.Task] = None
        self._stopped = False

    # ---- registration ----------------------------------------------------------

    def register_engine(
        self,
        engine: Engine,
        max_batch_size: int = 64,
    ) -> None:
        self.engines[engine.engine_id] = engine
        sched = LocalScheduler(engine, self.router.config)
        self.schedulers[engine.engine_id] = sched
        # Initial state snapshot so the router knows the engine exists.
        self.router.update_engine_state(self._snapshot_state(engine, max_batch_size))

    def _snapshot_state(self, engine: Engine, max_batch_size: int) -> EngineState:
        ks = engine.allocator.snapshot()
        sched = self.schedulers.get(engine.engine_id)
        return EngineState(
            engine_id=engine.engine_id,
            model_name=engine.partition.model_name,
            role=engine.role,
            waiting_queue_len=sched.queue_len() if sched else 0,
            active_sequences=engine.active_count(),
            current_batch_size=engine.active_count(),
            total_kv_blocks=ks["total_blocks"],
            free_kv_blocks=ks["free_blocks"],
            evictable_kv_blocks=ks["evictable_blocks"],
            # Tell the router which prefixes this engine has hot — so prefix
            # benefit scoring actually does something at runtime.
            prefix_cache=engine.prefix_cache_snapshot(),
            recent_ttft_ms=engine.recent_ttft_ms,
            recent_tpot_ms=engine.recent_tpot_ms,
            tensor_parallel_size=engine.partition.layout.tensor_parallel_size,
            pipeline_parallel_size=engine.partition.layout.pipeline_parallel_size,
            max_batch_size=max_batch_size,
        )

    # ---- lifecycle -------------------------------------------------------------

    async def start(self, report_period_s: float = 0.2) -> None:
        for eid, sched in self.schedulers.items():
            self._engine_tasks[eid] = asyncio.create_task(sched.run())
        self._reporter_task = asyncio.create_task(self._reporter_loop(report_period_s))

    async def stop(self) -> None:
        self._stopped = True
        for sched in self.schedulers.values():
            sched.stop()
        for task in self._engine_tasks.values():
            task.cancel()
        if self._reporter_task is not None:
            self._reporter_task.cancel()
        for task in list(self._engine_tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._reporter_task is not None:
            try:
                await self._reporter_task
            except asyncio.CancelledError:
                pass

    async def _reporter_loop(self, period_s: float) -> None:
        while not self._stopped:
            for engine in self.engines.values():
                self.router.update_engine_state(self._snapshot_state(engine, max_batch_size=engine.max_batch_size))
            await asyncio.sleep(period_s)

    # ---- request flow ----------------------------------------------------------

    async def submit(self, request: Request, disaggregated: bool = True) -> ServeResult:
        t0 = time.perf_counter()
        spilled: List[str] = []

        if not disaggregated:
            decision = self.router.route(request, EngineRole.COLOCATED)
            engine = self.engines[decision.engine_id]
            sched = self.schedulers[decision.engine_id]
            adm = await self._admit_until_ok(sched, request)
            spilled.extend(adm.spilled)
            t_prefill = time.perf_counter()
            await engine.admit_prefill(request)
            ttft_ms = (time.perf_counter() - t_prefill) * 1000.0
            tokens = await self._await_completion(engine, request.request_id)
            return ServeResult(
                request_id=request.request_id,
                output_tokens=tokens,
                prefill_engine=engine.engine_id,
                decode_engine=engine.engine_id,
                ttft_ms=ttft_ms,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                spilled=spilled,
            )

        # Disaggregated path: prefill -> KV transfer -> decode.
        prefill_decision, decode_decision = self.router.route_disaggregated(request)
        prefill_engine = self.engines[prefill_decision.engine_id]
        decode_engine = self.engines[decode_decision.engine_id]
        prefill_sched = self.schedulers[prefill_decision.engine_id]
        decode_sched = self.schedulers[decode_decision.engine_id]

        # 1. admit on prefill, run prompt forward.
        adm_p = await self._admit_until_ok(prefill_sched, request)
        spilled.extend(adm_p.spilled)
        t_prefill = time.perf_counter()
        first_token = await prefill_engine.admit_prefill(request)
        ttft_ms = (time.perf_counter() - t_prefill) * 1000.0

        # 2. handoff: copy KV state to decode engine.
        kv_state = await prefill_engine.export_kv(request.request_id)

        # 3. admit on decode, install KV.
        adm_d = await self._admit_until_ok(decode_sched, request)
        spilled.extend(adm_d.spilled)
        await decode_engine.admit_decode(request, first_token, kv_state)

        # 4. release prefill-side KV (decode is now authoritative).
        await prefill_engine.evict(request.request_id)

        # 5. wait for decode to finish and reap.
        tokens = await self._await_completion(decode_engine, request.request_id)
        return ServeResult(
            request_id=request.request_id,
            output_tokens=tokens,
            prefill_engine=prefill_engine.engine_id,
            decode_engine=decode_engine.engine_id,
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            spilled=spilled,
        )

    async def _admit_until_ok(
        self, sched: LocalScheduler, request: Request
    ) -> AdmitResult:
        """Admit, retrying on QUEUED until either admitted or rejected."""
        while True:
            r = await sched.admit(request)
            if r.outcome in (AdmitOutcome.ADMITTED, AdmitOutcome.SPILLED_AND_ADMITTED):
                return r
            if r.outcome == AdmitOutcome.REJECTED:
                raise RuntimeError(f"request {request.request_id} rejected: {r.detail}")
            # QUEUED — back off briefly and let the engine drain.
            await asyncio.sleep(0.01)

    async def _await_completion(self, engine: Engine, request_id: str) -> List[int]:
        """Block on the engine's completion event and consume the stashed output."""
        return await engine.wait_for(request_id)
