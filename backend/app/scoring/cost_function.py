"""cost_function.py — Composable scoring engine.

Assembles individual penalty components into a full :class:`ScoreBreakdown`.
Each scoring factor is an independent call so that individual terms can be
disabled, logged, or tuned without touching the others.

Public API:
    score_insertion(...)    → ScoreBreakdown   (insertion_cost_batching)
    score_dhobr_insertion(...) → (float, List[Coord])  (DHOBR batching)
    score_dhobr_new_batch(...) → (float, List[Coord])
    order_cost(...)         → float             (k-medoids)
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from app.optimizer.batching._types import BatchItem, BatchSummary, SmartBatch
from app.optimizer.batching.insertion import (
    route_node_distance,
    simulate_insertion,
)
from app.optimizer.batching.workload_balancer import picker_load_penalty
from app.optimizer.batching.similarity import jaccard_similarity
from app.optimizer.graph_model import GridGraph
from app.scoring.penalty_manager import (
    category_boost_items,
    delay_penalty_items,
    dominant_zone_from_items,
    fragility_penalty_order,
    item_similarity_penalty,
    same_zone_nearby_boost_items,
    urgency_penalty,
    workload_penalty,
    zone_mismatch_penalty_items,
    zone_set_from_items,
)
from app.scoring.scoring_models import ScoreBreakdown
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# insertion_cost_batching scorer
# ---------------------------------------------------------------------------

def score_insertion(
    order: Order,
    batch: List[Order],
    before_summary: BatchSummary,
    after_summary: BatchSummary,
    route_delta: float,
    order_zones: Set[str],
    order_categories: Set[str],
    order_flags: Dict[str, bool],
    batch_categories: Set[str],
    sku_category_lookup: Dict[str, str],
    batch_type: str,
    config: OptimizationConfig,
) -> ScoreBreakdown:
    """Compute the full insertion score for adding *order* to *batch*.

    Returns a :class:`ScoreBreakdown` that exposes each component for
    debugging and observability.  Callers use ``.total`` for the scalar score.
    """
    breakdown = ScoreBreakdown()

    # Route contribution
    breakdown.route_delta = config.route_cost_reweight_factor * route_delta

    # Urgency contribution
    urgency_window = (
        order.latest_pick_start_minutes
        if order.latest_pick_start_minutes is not None
        else float(order.due_time_minutes)
    )
    breakdown.urgency_penalty = (
        config.beta_due_time
        * urgency_penalty(after_summary.duration_seconds, urgency_window)
    )

    # Workload contribution
    breakdown.workload_penalty = (
        config.gamma_weight * workload_penalty(before_summary, config)
    )

    # Zone dissimilarity
    zone_similarity = (
        jaccard_similarity(order_zones, before_summary.zones)
        if before_summary.zones
        else 0.0
    )
    breakdown.zone_dissimilarity_penalty = (
        config.delta_similarity * (1.0 - zone_similarity)
    )

    # Fragility / temperature
    breakdown.fragility_penalty = fragility_penalty_order(
        order_flags, before_summary, order_zones, before_summary.zones, config
    )

    # Category similarity bonus (negative contribution)
    cat_set = {
        sku_category_lookup.get(sku, sku.split("-", 1)[0])
        for sku in batch_categories
    }
    category_similarity = jaccard_similarity(order_categories, cat_set)
    breakdown.category_similarity_bonus = (
        -config.similarity_batch_boost * category_similarity
    )

    # Priority bonus (negative contribution)
    breakdown.priority_bonus = -(
        float(order.priority or 0.0) * config.priority_score_weight
    )

    # Overflow penalty
    breakdown.overflow_penalty = (
        config.overflow_assignment_penalty
        if batch_type == "overflow"
        else 0.0
    )

    return breakdown


# ---------------------------------------------------------------------------
# DHOBR item-level scorers
# ---------------------------------------------------------------------------

def score_dhobr_insertion(
    grid: GridGraph,
    batch: SmartBatch,
    items: List[BatchItem],
    config: OptimizationConfig,
    picker_speed_mps: float,
) -> tuple[float, List[Coord]]:
    """Score inserting *items* into an existing DHOBR *batch*.

    Returns ``(score, candidate_route)``.  An infinite score signals that the
    zone constraint would be violated.
    """
    # Zone capacity hard constraint
    combined_zones = zone_set_from_items(batch.items + items)
    if len(combined_zones) > config.max_zones_per_batch:
        return float("inf"), list(batch.route)

    candidate_route, route_increase = simulate_insertion(grid, batch.route, items)
    route_after = route_node_distance(grid, candidate_route)

    score = (
        config.dhobr_route_weight * route_increase
        + zone_mismatch_penalty_items(batch, items, config)
        + config.dhobr_similarity_weight * item_similarity_penalty(grid, batch, items)
        + config.dhobr_picker_load_weight * picker_load_penalty(batch, config)
        + config.dhobr_delay_weight * delay_penalty_items(route_after, items, picker_speed_mps)
        - category_boost_items(batch, items, config)
        - same_zone_nearby_boost_items(grid, batch, items, config)
    )
    return score, candidate_route


def score_dhobr_new_batch(
    grid: GridGraph,
    start: Coord,
    end: Coord,
    items: List[BatchItem],
    config: OptimizationConfig,
    picker_speed_mps: float,
) -> tuple[float, List[Coord]]:
    """Score creating a new DHOBR batch seeded with *items*."""
    route = [start, end]
    route, route_increase = simulate_insertion(grid, route, items)
    score = (
        config.dhobr_route_weight * route_increase
        + config.dhobr_delay_weight
        * delay_penalty_items(route_node_distance(grid, route), items, picker_speed_mps)
        + config.dhobr_new_batch_bias
    )
    return score, route


# ---------------------------------------------------------------------------
# k-medoids order cost
# ---------------------------------------------------------------------------

def order_cost(
    feature: np.ndarray,
    medoid: np.ndarray,
    due_time_minutes: int,
    weight: float,
    order_categories: Set[str],
    batch_categories: Set[str],
    config: OptimizationConfig,
) -> float:
    """Compute the k-medoids assignment cost for one order.

    Combines spatial distance to medoid, urgency risk, weight pressure, and a
    category-similarity bonus.
    """
    distance_cost = float(np.linalg.norm(feature - medoid))
    due_risk = 1.0 / max(due_time_minutes, 1)
    similarity_bonus = (
        jaccard_similarity(order_categories, batch_categories)
        if batch_categories
        else 0.0
    )
    return (
        config.alpha_distance * distance_cost
        + config.beta_due_time * due_risk
        + config.gamma_weight * weight
        - config.delta_similarity * similarity_bonus
    )
