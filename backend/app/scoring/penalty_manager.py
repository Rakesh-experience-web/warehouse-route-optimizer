"""penalty_manager.py — Individual penalty calculators.

Each function computes exactly one penalty component and returns a scalar.
Callers (cost_function.py) compose these into a full ScoreBreakdown.

All functions are pure: they take immutable inputs and have no side-effects.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set, Tuple

from app.optimizer.batching._types import BatchItem, BatchSummary, SmartBatch
from app.optimizer.batching.similarity import jaccard_similarity
from app.optimizer.graph_model import GridGraph
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Order-level penalties (used in insertion_cost_batching)
# ---------------------------------------------------------------------------

def urgency_penalty(
    duration_seconds: float,
    urgency_window_minutes: float,
) -> float:
    """Return the lateness penalty for a batch with *duration_seconds*.

    Positive when the projected finish time exceeds *urgency_window_minutes*.
    """
    return max(0.0, (duration_seconds / 60.0) - urgency_window_minutes)


def workload_penalty(
    summary: BatchSummary,
    config: OptimizationConfig,
) -> float:
    """Return a combined capacity-pressure score for *summary*.

    Normalises order count, weight, and optionally volume against their
    respective ceilings so that the composite value is dimensionless.
    """
    pressure = summary.order_count / max(config.max_batch_size, 1)
    pressure += summary.total_weight / max(
        config.max_batch_weight, config.min_capacity_denominator
    )
    if config.max_batch_volume is not None and config.max_batch_volume > 0:
        pressure += summary.total_volume / config.max_batch_volume
    return pressure


def fragility_penalty_order(
    order_flags: Dict[str, bool],
    batch_summary: BatchSummary,
    order_zones: Set[str],
    batch_zones: Set[str],
    config: OptimizationConfig,
) -> float:
    """Return the fragility/temperature mismatch penalty for an order.

    Accumulates:
      - fragile-in-bulky-batch and bulky-in-fragile-batch combinations.
      - temperature-sensitive order routed through an ambient zone.
    """
    penalty = 0.0
    if order_flags.get("fragile") and batch_summary.bulky:
        penalty += config.fragile_bulky_penalty
    if order_flags.get("bulky") and batch_summary.fragile:
        penalty += config.fragile_bulky_penalty
    if (
        order_flags.get("temperature_sensitive")
        and batch_zones
        and "ambient" in batch_zones
        and order_zones
    ):
        penalty += config.temperature_zone_mismatch_penalty
    return penalty


# ---------------------------------------------------------------------------
# Item-level penalties (used in DHOBR / SmartBatch scoring)
# ---------------------------------------------------------------------------

def item_similarity_penalty(
    grid: GridGraph,
    batch: SmartBatch,
    items: List[BatchItem],
) -> float:
    """Return average travel distance from *items* to existing batch items.

    High values indicate the incoming items are spatially far from the
    existing batch, so grouping them together is costly.
    """
    if not batch.items or not items:
        return 0.0
    distances: List[float] = [
        grid.travel_cost(item.coord, existing.coord)
        for item in items
        for existing in batch.items
    ]
    return sum(distances) / len(distances) if distances else 0.0


def delay_penalty_items(
    route_distance_after: float,
    items: List[BatchItem],
    picker_speed_mps: float,
) -> float:
    """Return lateness penalty for a DHOBR batch after inserting *items*."""
    if not items:
        return 0.0
    estimated_finish_minutes = (
        route_distance_after / max(picker_speed_mps, 0.1)
    ) / 60.0
    due_time = min(item.due_time_minutes for item in items)
    return max(0.0, estimated_finish_minutes - due_time)


def category_boost_items(
    batch: SmartBatch,
    items: List[BatchItem],
    config: OptimizationConfig,
) -> float:
    """Return the category-similarity bonus for DHOBR insertion.

    Returned as a positive value; callers should subtract it from the total
    score.
    """
    if not batch.items or not items:
        return 0.0
    batch_categories = {item.category for item in batch.items}
    item_categories = {item.category for item in items}
    return config.advanced_category_boost_weight * jaccard_similarity(
        batch_categories, item_categories
    )


def dominant_zone_from_items(items: List[BatchItem]) -> str | None:
    """Return the most frequently occurring zone among *items*, or None."""
    if not items:
        return None
    return Counter(item.zone for item in items).most_common(1)[0][0]


def zone_set_from_items(items: List[BatchItem]) -> Set[str]:
    """Return the set of zones referenced by *items*."""
    return {item.zone for item in items}


def zone_mismatch_penalty_items(
    batch: SmartBatch,
    items: List[BatchItem],
    config: OptimizationConfig,
) -> float:
    """Return the zone-mismatch penalty for DHOBR insertion.

    Zero when the batch has no dominant zone or *items* all share it.
    """
    dominant_zone = dominant_zone_from_items(batch.items)
    if dominant_zone is None:
        return 0.0
    incoming_zones = zone_set_from_items(items)
    if incoming_zones == {dominant_zone}:
        return 0.0
    return config.zone_mismatch_penalty_weight * len(
        incoming_zones - {dominant_zone}
    )


def same_zone_nearby_boost_items(
    grid: GridGraph,
    batch: SmartBatch,
    items: List[BatchItem],
    config: OptimizationConfig,
) -> float:
    """Return a proximity bonus when *items* share the batch's dominant zone.

    Higher spatial proximity (lower average distance) results in a larger
    bonus.  Returned as a positive value; callers should subtract it.
    """
    dominant_zone = dominant_zone_from_items(batch.items)
    if (
        dominant_zone is None
        or not items
        or any(item.zone != dominant_zone for item in items)
    ):
        return 0.0
    similarity = item_similarity_penalty(grid, batch, items)
    return config.same_zone_nearby_boost_weight / max(1.0, similarity)
