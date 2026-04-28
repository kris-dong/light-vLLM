"""
Per-engine local scheduler — admit / spill / queue.

Decision flow (mirrors the user's requirements):

    new request arrives
        |
        v
   does adding it overflow KV (or break SLO)?
        |
   yes  | no
   |    |
   |    v
   |   ADMIT   -> add to current iteration batch
   |
   v
  any spill candidate (lower priority + preemptible)?
        |
   yes  | no
   |    |
   |    v
   |   QUEUE   -> wait until a finishing seq frees blocks
   |
   v
  SPILL victim, ADMIT new request

Iteration loop:
    after each forward step, completed sequences free KV; the scheduler
    re-evaluates the wait queue and the (possibly larger) free pool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .engine import Engine
from .router import Request, RouterConfig


class AdmitOutcome(str, Enum):
    ADMITTED = "admitted"
    QUEUED = "queued"
    SPILLED_AND_ADMITTED = "spilled_and_admitted"
    REJECTED = "rejected"


@dataclass
class AdmitResult:
    outcome: AdmitOutcome
    spilled: List[str]
    detail: str = ""


class LocalScheduler:
    """One scheduler per engine. Drives admission and the iteration step loop."""

    def __init__(self, engine: Engine, router_config: RouterConfig) -> None:
        self.engine = engine
        self.config = router_config
        self._wait_queue: "asyncio.Queue[Tuple[Request, asyncio.Future]]" = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()

    # ---- admission -------------------------------------------------------------

    def _blocks_needed(self, request: Request) -> int:
        if self.config.reserve_expected_decode:
            n = request.prompt_len + request.predicted_new_tokens
        else:
            n = request.prompt_len + request.max_new_tokens
        return self.engine.allocator.blocks_for_tokens(n)

    def _slo_admissible(self, request: Request) -> bool:
        """Reject if the engine's recent TTFT/TPOT already exceeds the SLO."""
        if (request.ttft_slo_ms is not None
                and self.engine.recent_ttft_ms > request.ttft_slo_ms * 1.5):
            return False
        if (request.tpot_slo_ms is not None
                and self.engine.recent_tpot_ms > request.tpot_slo_ms * 1.5):
            return False
        return True

    async def admit(self, request: Request) -> AdmitResult:
        """Try to admit ``request`` immediately; spill or queue if needed."""
        async with self._lock:
            if not self._slo_admissible(request):
                return AdmitResult(AdmitOutcome.REJECTED, [], "SLO already violated on this engine")

            need = self._blocks_needed(request)
            margin = self.config.min_free_kv_blocks_after_admit
            free = self.engine.allocator.free_blocks

            if free - need >= margin:
                self.engine.waiting_queue_len = self._wait_queue.qsize()
                return AdmitResult(AdmitOutcome.ADMITTED, [])

            # Need to free (need + margin - free) more blocks.
            shortfall = need + margin - free
            victims = self.engine.allocator.select_spill_victims(
                blocks_needed=shortfall,
                protect_request_ids=[request.request_id],
            )
            if not victims:
                # Queue and let the iteration loop retry once seqs finish.
                fut: "asyncio.Future[AdmitResult]" = asyncio.get_event_loop().create_future()
                await self._wait_queue.put((request, fut))
                self.engine.waiting_queue_len = self._wait_queue.qsize()
                return AdmitResult(AdmitOutcome.QUEUED, [], "no spill candidate; queued")

            for rid in victims:
                await self.engine.evict(rid)
            return AdmitResult(AdmitOutcome.SPILLED_AND_ADMITTED, list(victims))

    # ---- iteration loop --------------------------------------------------------

    async def run(self, step_period_s: float = 0.0) -> None:
        """
        Drive iteration-level scheduling. Each tick:
            1. step() the engine -> finish a token per active seq.
            2. reap finished seqs (freeing KV).
            3. retry queued requests against the freshly-freed pool.

        Caller awaits this in its own task; ``stop()`` ends the loop.
        """
        while not self._stop.is_set():
            if self.engine.active_count() == 0 and self._wait_queue.empty():
                await asyncio.sleep(step_period_s or 0.001)
                continue

            finished = await self.engine.step()
            for rid in finished:
                await self.engine.reap(rid)

            await self._drain_wait_queue()

            if step_period_s > 0:
                await asyncio.sleep(step_period_s)

    async def _drain_wait_queue(self) -> None:
        """Try queued requests against freshly available KV."""
        if self._wait_queue.empty():
            return
        deferred: List[Tuple[Request, asyncio.Future]] = []
        async with self._lock:
            free = self.engine.allocator.free_blocks
            margin = self.config.min_free_kv_blocks_after_admit
            while not self._wait_queue.empty():
                req, fut = self._wait_queue.get_nowait()
                if fut.done():
                    continue
                need = self._blocks_needed(req)
                if free - need >= margin:
                    free -= need
                    fut.set_result(AdmitResult(AdmitOutcome.ADMITTED, [], "drained from wait queue"))
                else:
                    deferred.append((req, fut))
            for item in deferred:
                await self._wait_queue.put(item)
            self.engine.waiting_queue_len = self._wait_queue.qsize()

    def stop(self) -> None:
        self._stop.set()

    # ---- inspection ------------------------------------------------------------

    def queue_len(self) -> int:
        return self._wait_queue.qsize()
