from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import math
import time


class EngineRole(str, Enum):
    COLOCATED = "colocated"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class Request:
    request_id: str
    model_name: str
    prompt_tokens: Sequence[int]

    # Upper bound supplied by user / API.
    max_new_tokens: int

    # Optional better prediction from historical statistics / request class.
    expected_new_tokens: Optional[int] = None

    # SLOs in milliseconds.
    ttft_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None

    # Higher means more important.
    priority: float = 1.0

    # For prefix-aware routing. Number of prompt tokens used for cache lookup.
    prefix_len_for_cache: int = 256

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def predicted_new_tokens(self) -> int:
        # Conservative but not worst-case by default.
        if self.expected_new_tokens is not None:
            return min(self.expected_new_tokens, self.max_new_tokens)
        return max(1, int(0.5 * self.max_new_tokens))

    @property
    def total_predicted_tokens(self) -> int:
        return self.prompt_len + self.predicted_new_tokens

    def prefix_hash(self) -> str:
        prefix = self.prompt_tokens[: self.prefix_len_for_cache]
        b = ",".join(map(str, prefix)).encode("utf-8")
        return hashlib.sha256(b).hexdigest()


@dataclass
class EngineState:
    engine_id: str
    model_name: str
    role: EngineRole

    # Current scheduling state.
    waiting_queue_len: int
    active_sequences: int
    current_batch_size: int

    # KV block accounting.
    total_kv_blocks: int
    free_kv_blocks: int
    evictable_kv_blocks: int = 0

    # Prefix cache summary.
    # Maps prefix hash -> number of reusable KV blocks or a rough hit score.
    prefix_cache: Dict[str, int] = field(default_factory=dict)

    # Observed recent performance.
    recent_ttft_ms: float = 0.0
    recent_tpot_ms: float = 0.0

    # Hardware / parallelism metadata.
    gpu_group_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Capacity estimates.
    max_batch_size: int = 128

    # Used to avoid stale worker states.
    last_report_time_s: float = field(default_factory=time.time)

    def supports(self, request: Request, role: EngineRole) -> bool:
        if self.model_name != request.model_name:
            return False

        if role == EngineRole.PREFILL:
            return self.role in (EngineRole.PREFILL, EngineRole.COLOCATED)

        if role == EngineRole.DECODE:
            return self.role in (EngineRole.DECODE, EngineRole.COLOCATED)

        return self.role == EngineRole.COLOCATED


@dataclass
class RouterConfig:
    # KV layout.
    kv_block_tokens: int = 16

    # Safety margin to avoid admitting requests that immediately cause pressure.
    min_free_kv_blocks_after_admit: int = 64

    # If true, use expected length instead of max_new_tokens for decode reservation.
    reserve_expected_decode: bool = True

    # Cost model coefficients.
    prefill_ms_per_token: float = 0.015
    decode_ms_per_token_step: float = 0.05

    # Scoring weights.
    w_queue: float = 1.0
    w_prefill: float = 1.0
    w_decode: float = 1.0
    w_kv_pressure: float = 2.0
    w_prefix_hit: float = 1.5
    w_slo_risk: float = 3.0
    w_batch_pressure: float = 0.7
    w_staleness: float = 10.0


@dataclass
class RouteDecision:
    request_id: str
    engine_id: str
    role: EngineRole
    score: float
    reason: Dict[str, float]


class LLMRouter:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.engines: Dict[str, EngineState] = {}

    def update_engine_state(self, state: EngineState) -> None:
        state.last_report_time_s = time.time()
        self.engines[state.engine_id] = state

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.config.kv_block_tokens)

    def _initial_kv_blocks_needed(self, request: Request) -> int:
        """
        Blocks needed after prefill.
        Prompt KV must fit before decode starts.
        """
        return self._blocks_for_tokens(request.prompt_len)

    def _predicted_total_kv_blocks_needed(self, request: Request) -> int:
        """
        Approximate future KV demand.
        This is used for risk scoring, not necessarily hard reservation.
        """
        if self.config.reserve_expected_decode:
            total_tokens = request.prompt_len + request.predicted_new_tokens
        else:
            total_tokens = request.prompt_len + request.max_new_tokens
        return self._blocks_for_tokens(total_tokens)

    def _prefix_hit_blocks(self, engine: EngineState, request: Request) -> int:
        return engine.prefix_cache.get(request.prefix_hash(), 0)

    def _has_kv_capacity_for_admission(
        self,
        engine: EngineState,
        request: Request,
        role: EngineRole,
    ) -> bool:
        """
        For prefill, at least prompt KV must fit.
        For decode, predicted total KV should preferably fit.
        """
        if role == EngineRole.PREFILL: 
            needed = self._initial_kv_blocks_needed(request)
        else:
            needed = self._predicted_total_kv_blocks_needed(request)

        effective_free = engine.free_kv_blocks + engine.evictable_kv_blocks
        return (
            effective_free - needed
            >= self.config.min_free_kv_blocks_after_admit
        )

    def _estimate_prefill_ms(self, request: Request, engine: EngineState) -> float:
        """
        Simple model. Real systems would fit this from profiling.
        Longer prompt -> higher prefill cost.
        More tensor parallelism may reduce compute time but add communication.
        """
        base = request.prompt_len * self.config.prefill_ms_per_token
        parallel_discount = 1.0 / max(1, engine.tensor_parallel_size) ** 0.7
        comm_penalty = 1.0 + 0.05 * max(0, engine.tensor_parallel_size - 1)
        return base * parallel_discount * comm_penalty

    def _estimate_decode_pressure_ms(self, request: Request, engine: EngineState) -> float:
        """
        Decode pressure is affected by active batch and predicted generation length.
        This is not total latency; it is a routing pressure score.
        """
        predicted_steps = request.predicted_new_tokens
        batch_factor = 1.0 + engine.current_batch_size / max(1, engine.max_batch_size)
        context_factor = math.log2(max(2, request.total_predicted_tokens))
        return (
            predicted_steps
            * self.config.decode_ms_per_token_step
            * batch_factor
            * context_factor
        )

    def _estimate_queue_delay_ms(self, engine: EngineState) -> float:
        """
        Coarse queue model.
        A real router would use moving averages and per-engine scheduler state.
        """
        recent_step = max(engine.recent_tpot_ms, 1.0)
        return engine.waiting_queue_len * recent_step

    def _kv_pressure_score(self, engine: EngineState, request: Request) -> float:
        needed = self._predicted_total_kv_blocks_needed(request)
        effective_free = engine.free_kv_blocks + engine.evictable_kv_blocks

        if effective_free <= 0:
            return float("inf")

        after = effective_free - needed
        if after <= 0:
            return float("inf")

        # Higher when closer to full.
        used_ratio_after = 1.0 - after / engine.total_kv_blocks
        return max(0.0, used_ratio_after)

    def _prefix_benefit_score(self, engine: EngineState, request: Request) -> float:
        """
        Higher reusable prefix blocks should reduce routing score.
        """
        hit_blocks = self._prefix_hit_blocks(engine, request)
        prompt_blocks = max(1, self._initial_kv_blocks_needed(request))
        return min(1.0, hit_blocks / prompt_blocks)

    def _slo_risk_score(
        self,
        engine: EngineState,
        request: Request,
        role: EngineRole,
        predicted_queue_ms: float,
        predicted_prefill_ms: float,
    ) -> float:
        risk = 0.0

        if role in (EngineRole.PREFILL, EngineRole.COLOCATED):
            if request.ttft_slo_ms is not None:
                predicted_ttft = (
                    predicted_queue_ms
                    + predicted_prefill_ms
                    + max(engine.recent_ttft_ms, 0.0)
                )
                risk += max(0.0, predicted_ttft / request.ttft_slo_ms - 1.0)

        if role in (EngineRole.DECODE, EngineRole.COLOCATED):
            if request.tpot_slo_ms is not None and engine.recent_tpot_ms > 0:
                risk += max(0.0, engine.recent_tpot_ms / request.tpot_slo_ms - 1.0)

        # Priority reduces effective risk.
        return risk / max(request.priority, 1e-6)

    def _batch_pressure_score(self, engine: EngineState) -> float:
        return engine.current_batch_size / max(1, engine.max_batch_size)

    def _staleness_score(self, engine: EngineState) -> float:
        age_s = time.time() - engine.last_report_time_s
        # No penalty for very fresh state; increasing penalty after 2 seconds.
        return max(0.0, age_s - 2.0)

    def _score_engine(
        self,
        engine: EngineState,
        request: Request,
        role: EngineRole,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if not engine.supports(request, role):
            return None

        if not self._has_kv_capacity_for_admission(engine, request, role):
            return None

        queue_ms = self._estimate_queue_delay_ms(engine)
        prefill_ms = self._estimate_prefill_ms(request, engine)
        decode_pressure_ms = self._estimate_decode_pressure_ms(request, engine)
        kv_pressure = self._kv_pressure_score(engine, request)
        prefix_benefit = self._prefix_benefit_score(engine, request)
        slo_risk = self._slo_risk_score(
            engine=engine,
            request=request,
            role=role,
            predicted_queue_ms=queue_ms,
            predicted_prefill_ms=prefill_ms,
        )
        batch_pressure = self._batch_pressure_score(engine)
        staleness = self._staleness_score(engine)

        if math.isinf(kv_pressure):
            return None

        c = self.config

        score = (
            c.w_queue * queue_ms
            + c.w_prefill * prefill_ms
            + c.w_decode * decode_pressure_ms
            + c.w_kv_pressure * kv_pressure
            - c.w_prefix_hit * prefix_benefit
            + c.w_slo_risk * slo_risk
            + c.w_batch_pressure * batch_pressure
            + c.w_staleness * staleness
        )

        reason = {
            "queue_ms": queue_ms,
            "prefill_ms": prefill_ms,
            "decode_pressure_ms": decode_pressure_ms,
            "kv_pressure": kv_pressure,
            "prefix_benefit": prefix_benefit,
            "slo_risk": slo_risk,
            "batch_pressure": batch_pressure,
            "staleness": staleness,
        }

        return score, reason

    def route(
        self,
        request: Request,
        role: EngineRole = EngineRole.COLOCATED,
    ) -> RouteDecision:
        candidates: List[RouteDecision] = []

        for engine in self.engines.values():
            scored = self._score_engine(engine, request, role)
            if scored is None:
                continue

            score, reason = scored
            candidates.append(
                RouteDecision(
                    request_id=request.request_id,
                    engine_id=engine.engine_id,
                    role=role,
                    score=score,
                    reason=reason,
                )
            )

        if not candidates:
            raise RuntimeError(
                f"No feasible engine for request={request.request_id}, "
                f"model={request.model_name}, role={role.value}"
            )

        return min(candidates, key=lambda x: x.score)

    def route_disaggregated(
        self,
        request: Request,
    ) -> Tuple[RouteDecision, RouteDecision]:
        """
        First choose a prefill worker, then choose a decode worker.
        In a real system, the decode choice may happen after prefill because
        the prefill output/KV size and first-token timing are then known.
        """
        prefill_decision = self.route(request, EngineRole.PREFILL)
        decode_decision = self.route(request, EngineRole.DECODE)
        return prefill_decision, decode_decision


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    router = LLMRouter(
        RouterConfig(
            kv_block_tokens=16,
            min_free_kv_blocks_after_admit=32,
            prefill_ms_per_token=0.012,
            decode_ms_per_token_step=0.04,
        )
    )

    # Example tokenized request.
    prompt_tokens = [101, 42, 77, 91, 13, 52] * 200  # length 1200

    req = Request(
        request_id="req-001",
        model_name="llama-70b",
        prompt_tokens=prompt_tokens,
        max_new_tokens=512,
        expected_new_tokens=160,
        ttft_slo_ms=800.0,
        tpot_slo_ms=80.0,
        priority=1.0,
    )

    prefix_hash = req.prefix_hash()

    # Register some colocated engines.
    router.update_engine_state(
        EngineState(
            engine_id="engine-a",
            model_name="llama-70b",
            role=EngineRole.COLOCATED,
            waiting_queue_len=4,
            active_sequences=48,
            current_batch_size=48,
            total_kv_blocks=10000,
            free_kv_blocks=3000,
            evictable_kv_blocks=500,
            prefix_cache={prefix_hash: 60},
            recent_ttft_ms=500.0,
            recent_tpot_ms=45.0,
            tensor_parallel_size=4,
            max_batch_size=128,
        )
    )

    router.update_engine_state(
        EngineState(
            engine_id="engine-b",
            model_name="llama-70b",
            role=EngineRole.COLOCATED,
            waiting_queue_len=1,
            active_sequences=96,
            current_batch_size=96,
            total_kv_blocks=10000,
            free_kv_blocks=1200,
            evictable_kv_blocks=100,
            prefix_cache={},
            recent_ttft_ms=300.0,
            recent_tpot_ms=70.0,
            tensor_parallel_size=4,
            max_batch_size=128,
        )
    )

    decision = router.route(req, EngineRole.COLOCATED)

    print("Colocated routing decision:")
    print(decision)

    # Register disaggregated workers.
    router.update_engine_state(
        EngineState(
            engine_id="prefill-0",
            model_name="llama-70b",
            role=EngineRole.PREFILL,
            waiting_queue_len=0,
            active_sequences=8,
            current_batch_size=8,
            total_kv_blocks=8000,
            free_kv_blocks=5000,
            prefix_cache={prefix_hash: 60},
            recent_ttft_ms=250.0,
            recent_tpot_ms=0.0,
            tensor_parallel_size=8,
            max_batch_size=64,
        )
    )

    router.update_engine_state(
        EngineState(
            engine_id="decode-0",
            model_name="llama-70b",
            role=EngineRole.DECODE,
            waiting_queue_len=2,
            active_sequences=64,
            current_batch_size=64,
            total_kv_blocks=16000,
            free_kv_blocks=7000,
            evictable_kv_blocks=1000,
            prefix_cache={},
            recent_ttft_ms=0.0,
            recent_tpot_ms=38.0,
            tensor_parallel_size=4,
            max_batch_size=160,
        )
    )

    prefill_route, decode_route = router.route_disaggregated(req)

    print("\nDisaggregated routing decision:")
    print("prefill:", prefill_route)
    print("decode: ", decode_route)
