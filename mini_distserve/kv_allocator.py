"""
Paged KV-block allocator (vLLM-style, distilled).

A request's KV is stored as a list of fixed-size token blocks. The allocator
hands out free block ids, supports eviction of preemptible owners, and tracks
a per-block reference count so multiple owners can share the same blocks for
an identical prefix.

How prefix sharing actually works end-to-end (wired by the Engine):

  1. After a fresh prefill, the Engine takes the first N blocks of the
     request's allocation (where N covers a prefix-hash window) and registers
     them under a synthetic owner id ``__prefix__:<hash>`` via
     ``allocate(..., share_block_ids=those_block_ids, preemptible=False)``.
     This bumps each block's refcount, pinning them past the original
     request's lifetime.

  2. On a future ``allocate(..., share_block_ids=cached_block_ids)`` for a
     request whose prompt prefix matches, the cached block ids are reused
     (refcount +1) and only suffix blocks are pulled from the free pool —
     a real KV-memory saving plus a real prefill-compute saving (the engine
     also passes the cached KV tensors to the backend so it forwards only
     the suffix).

  3. ``free`` decrements refcount on every owned block; a block returns to
     the free pool only when refcount hits 0 — i.e., when both the request
     and any prefix-cache owner have released it.

  4. The Engine evicts old prefixes (LRU) by calling ``free`` on the synthetic
     owner; if no live request still references those blocks they become free.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class KVAllocation:
    """Per-request handle. The scheduler uses block_ids; the engine consumes them."""

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    # True when the request can be preempted to reclaim its KV.
    preemptible: bool = True
    # Lower means easier to evict.
    priority: float = 1.0


class KVAllocator:
    def __init__(self, total_blocks: int, block_size_tokens: int):
        if total_blocks <= 0 or block_size_tokens <= 0:
            raise ValueError("total_blocks and block_size_tokens must be positive")
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens

        self._free: Set[int] = set(range(total_blocks))
        self._owners: Dict[str, KVAllocation] = {}
        # Insertion-ordered so eviction picks the oldest preemptible owner first
        # within a given priority tier.
        self._owner_order: "OrderedDict[str, None]" = OrderedDict()
        # Reference count per block id for prefix sharing.
        self._refcount: Dict[int, int] = {}
        self._lock = threading.Lock()

    # ---- sizing ----------------------------------------------------------------

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size_tokens)

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def used_blocks(self) -> int:
        with self._lock:
            return self.total_blocks - len(self._free)

    def evictable_blocks(self) -> int:
        with self._lock:
            return sum(
                len(a.block_ids) for a in self._owners.values() if a.preemptible
            )

    # ---- admission -------------------------------------------------------------

    def can_admit(self, num_tokens: int, safety_margin_blocks: int = 0) -> bool:
        need = self.blocks_for_tokens(num_tokens)
        with self._lock:
            return len(self._free) - need >= safety_margin_blocks

    # ---- alloc/free ------------------------------------------------------------

    def allocate(
        self,
        request_id: str,
        num_tokens: int,
        preemptible: bool = True,
        priority: float = 1.0,
        share_block_ids: Optional[Iterable[int]] = None,
    ) -> KVAllocation:
        """
        Allocate enough blocks for ``num_tokens`` tokens. ``share_block_ids``
        lets a new request reuse blocks already owned by another (prefix cache
        hit); their refcount is bumped so they survive the original owner's
        free.
        """
        need = self.blocks_for_tokens(num_tokens)
        shared = list(share_block_ids or [])
        fresh_needed = max(0, need - len(shared))

        with self._lock:
            if request_id in self._owners:
                raise ValueError(f"duplicate request_id {request_id!r}")
            if len(self._free) < fresh_needed:
                raise RuntimeError(
                    f"out of KV: need {fresh_needed} fresh blocks, free={len(self._free)}"
                )

            block_ids: List[int] = list(shared)
            for bid in shared:
                self._refcount[bid] = self._refcount.get(bid, 0) + 1
            for _ in range(fresh_needed):
                bid = self._free.pop()
                block_ids.append(bid)
                self._refcount[bid] = 1

            alloc = KVAllocation(
                request_id=request_id,
                block_ids=block_ids,
                preemptible=preemptible,
                priority=priority,
            )
            self._owners[request_id] = alloc
            self._owner_order[request_id] = None
            return alloc

    def extend(self, request_id: str, extra_tokens: int) -> List[int]:
        """Append blocks to an existing allocation (used during decode growth)."""
        if extra_tokens <= 0:
            return []
        with self._lock:
            alloc = self._owners.get(request_id)
            if alloc is None:
                raise KeyError(request_id)
            current_tokens = len(alloc.block_ids) * self.block_size_tokens
            new_total = current_tokens + extra_tokens
            new_blocks_total = self.blocks_for_tokens(new_total)
            extra = new_blocks_total - len(alloc.block_ids)
            if extra <= 0:
                return []
            if len(self._free) < extra:
                raise RuntimeError(
                    f"out of KV on extend: need {extra}, free={len(self._free)}"
                )
            added: List[int] = []
            for _ in range(extra):
                bid = self._free.pop()
                alloc.block_ids.append(bid)
                self._refcount[bid] = 1
                added.append(bid)
            return added

    def free(self, request_id: str) -> int:
        """Drop refcount on this owner's blocks; return blocks actually returned to free pool."""
        with self._lock:
            alloc = self._owners.pop(request_id, None)
            if alloc is None:
                return 0
            self._owner_order.pop(request_id, None)
            returned = 0
            for bid in alloc.block_ids:
                rc = self._refcount.get(bid, 0) - 1
                if rc <= 0:
                    self._refcount.pop(bid, None)
                    self._free.add(bid)
                    returned += 1
                else:
                    self._refcount[bid] = rc
            return returned

    # ---- spilling --------------------------------------------------------------

    def select_spill_victims(
        self,
        blocks_needed: int,
        protect_request_ids: Iterable[str] = (),
    ) -> List[str]:
        """Pick preemptible owners (low-priority first, then oldest) until we'd reclaim
        ``blocks_needed`` blocks. Does NOT free them; caller decides whether to commit.
        """
        protect = set(protect_request_ids)
        with self._lock:
            candidates = [
                self._owners[rid]
                for rid in self._owner_order
                if rid not in protect and self._owners[rid].preemptible
            ]
        candidates.sort(key=lambda a: (a.priority, self._owner_order_index(a.request_id)))
        victims: List[str] = []
        reclaimed = 0
        for a in candidates:
            if reclaimed >= blocks_needed:
                break
            victims.append(a.request_id)
            reclaimed += len(a.block_ids)
        if reclaimed < blocks_needed:
            return []  # not enough even with full eviction; caller should reject/queue
        return victims

    def _owner_order_index(self, rid: str) -> int:
        # Best-effort positional rank for tiebreaking; cheap on small batches.
        for i, k in enumerate(self._owner_order):
            if k == rid:
                return i
        return 1 << 30

    # ---- introspection ---------------------------------------------------------

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_blocks": self.total_blocks,
                "free_blocks": len(self._free),
                "used_blocks": self.total_blocks - len(self._free),
                "owners": len(self._owners),
                "evictable_blocks": sum(
                    len(a.block_ids) for a in self._owners.values() if a.preemptible
                ),
            }

    def get_block_ids(self, request_id: str) -> List[int]:
        with self._lock:
            alloc = self._owners.get(request_id)
            return list(alloc.block_ids) if alloc else []
