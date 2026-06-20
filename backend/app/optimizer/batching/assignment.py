"""assignment.py — Order-sorting and batch-finalisation utilities.

Responsibilities:
  - Determine the effective number of batches to create.
  - Sort orders for optimal assignment ordering.
  - Finalise a raw batch list into a :class:`BatchAssignment`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from app.optimizer.batching._types import BatchAssignment
from app.optimizer.batching.similarity import average_pairwise_similarity, order_category_set
from app.optimizer.feature_engineering import order_weight
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Effective batch count
# ---------------------------------------------------------------------------

def effective_batch_count(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> int:
    """Compute the optimal number of batches for *orders*.

    When dynamic batching is disabled the configured ``batch_count`` is used
    directly (capped to the order count).  Otherwise the count adapts based on:

    1. A capacity floor driven by weight and order-count limits.
    2. A similarity target that increases when orders are heterogeneous so that
       similar orders can cluster together.

    The final value is always in the range ``[1, len(orders)]``.
    """
    if not orders:
        return 0

    min_batches_by_size = max(1, math.ceil(len(orders) / max(config.max_batch_size, 1)))
    total_weight = sum(order_weight(order) for order in orders)
    min_batches_by_weight = max(
        1,
        math.ceil(total_weight / max(config.max_batch_weight, config.min_capacity_denominator)),
    )
    capacity_floor = max(min_batches_by_size, min_batches_by_weight)

    if not config.dynamic_batching_enabled:
        return min(max(config.batch_count, 1), len(orders))

    desired_upper = min(
        max(config.batch_count, capacity_floor, 1),
        config.employee_count,
        len(orders),
    )
    if desired_upper == 1:
        return 1

    order_category_sets: List[Set[str]] = [
        order_category_set(o, sku_category_lookup) for o in orders
    ]
    avg_similarity = average_pairwise_similarity(order_category_sets)
    dissimilarity = max(0.0, 1.0 - avg_similarity)
    # Dynamic batching should adapt the total number of planned batches, not
    # cap it to simultaneous pickers. Higher order volumes still need multiple
    # execution waves even when only a few pickers are active concurrently.
    similarity_target = 1 + int(math.floor(dissimilarity * (desired_upper - 1)))
    k = max(capacity_floor, similarity_target)
    return min(max(k, 1), config.employee_count, len(orders))


# ---------------------------------------------------------------------------
# Order sorting
# ---------------------------------------------------------------------------

def sorted_orders_for_assignment(orders: List[Order]) -> List[Order]:
    """Sort *orders* for insertion into batches.

    Primary sort key is the urgency window (``latest_pick_start_minutes`` or
    ``due_time_minutes``).  Secondary keys break ties by due time, then by
    descending priority, then by creation epoch.
    """

    def key(order: Order) -> tuple[float, int, float, int]:
        urgency_floor = (
            order.latest_pick_start_minutes
            if order.latest_pick_start_minutes is not None
            else order.due_time_minutes
        )
        priority = float(order.priority or 0.0)
        return (float(urgency_floor), order.due_time_minutes, -priority, order.created_at_epoch)

    return sorted(orders, key=key)


# ---------------------------------------------------------------------------
# K-medoids centroid initialisation
# ---------------------------------------------------------------------------

def init_medoids(features: np.ndarray, k: int) -> List[int]:
    """Select *k* initial medoid indices using the maximin heuristic.

    When ``len(features) <= k`` all indices are returned as medoids.
    """
    if len(features) <= k:
        return list(range(len(features)))
    chosen = [0]
    while len(chosen) < k:
        d2 = np.array(
            [
                min(np.sum((features[i] - features[c]) ** 2) for c in chosen)
                for i in range(len(features))
            ],
            dtype=np.float64,
        )
        nxt = int(np.argmax(d2))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return chosen[:k]


# ---------------------------------------------------------------------------
# Batch finalisation
# ---------------------------------------------------------------------------

def finalize_assignment(
    batches: List[List[Order]],
    batch_names: List[str],
    batch_types: List[str],
    *,
    notes: List[str] | None = None,
    exception_order_ids: List[str] | None = None,
) -> BatchAssignment:
    """Convert raw batch lists into a :class:`BatchAssignment`.

    Empty batches are dropped.  The ``labels`` mapping is built by assigning
    each order's ``order_id`` to the index of its first non-empty batch.
    """
    non_empty_batches: List[List[Order]] = []
    non_empty_names: List[str] = []
    non_empty_types: List[str] = []
    labels: Dict[str, int] = {}

    for batch, batch_name, batch_type in zip(batches, batch_names, batch_types):
        if not batch:
            continue
        new_idx = len(non_empty_batches)
        non_empty_batches.append(batch)
        non_empty_names.append(batch_name)
        non_empty_types.append(batch_type)
        for order in batch:
            if order.order_id not in labels:
                labels[order.order_id] = new_idx

    return BatchAssignment(
        batches=non_empty_batches,
        labels=labels,
        batch_names=non_empty_names,
        batch_types=non_empty_types,
        notes=list(notes or []),
        exception_order_ids=list(exception_order_ids or []),
    )
