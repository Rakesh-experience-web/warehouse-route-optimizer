"""constraints.py — Isolated constraint validators for batch feasibility.

Each validator is a pure function that accepts a summary and config and returns
a structured result.  New constraints can be added without touching the
assignment algorithms.

Exported public API:
    batch_constraint_violations(summary, config) -> List[str]
    batch_feasible(...) -> bool
    zone_limit_feasible(batch_orders, candidate, zone_lookup, config) -> bool
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from app.optimizer.batching._types import BatchSummary
from app.optimizer.batching.similarity import order_category_set
from app.optimizer.feature_engineering import order_zone_set
from app.optimizer.graph_model import GridGraph
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Hard-constraint violation checker
# ---------------------------------------------------------------------------

def batch_constraint_violations(
    summary: BatchSummary,
    config: OptimizationConfig,
) -> List[str]:
    """Return a list of violated hard constraint names for *summary*.

    An empty list means the batch is feasible under all hard constraints.

    Violations checked:
      - ``max_batch_size``: order count ceiling.
      - ``max_batch_weight``: weight ceiling.
      - ``max_batch_volume``: optional volume ceiling.
      - ``max_shelf_visits_per_picker``: number of distinct pick targets.
      - ``max_batch_duration_seconds``: optional route duration ceiling.
    """
    violations: List[str] = []

    if summary.order_count > config.max_batch_size:
        violations.append("max_batch_size")

    if summary.total_units > getattr(config, "max_batch_units", 25):
        violations.append("max_batch_units")

    if summary.total_weight > config.max_batch_weight:
        violations.append("max_batch_weight")

    if (
        config.max_batch_volume is not None
        and summary.total_volume > config.max_batch_volume
    ):
        violations.append("max_batch_volume")

    if summary.target_count > config.max_shelf_visits_per_picker:
        violations.append("max_shelf_visits_per_picker")

    if (
        config.max_batch_duration_seconds is not None
        and summary.duration_seconds > config.max_batch_duration_seconds
    ):
        violations.append("max_batch_duration_seconds")

    return violations


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------

def batch_feasible(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    config: OptimizationConfig,
    grid: GridGraph,
    start: Coord,
    end: Coord,
    picker_speed_mps: float,
) -> bool:
    """Return True when *batch_orders* satisfies all hard constraints.

    Builds a :class:`BatchSummary` internally using the route-estimate proxy
    so that the full constraint set (including duration) is evaluated.
    """
    # Import here to avoid circular dependency at module load time.
    from app.optimizer.batching._summary import summarize_batch

    summary = summarize_batch(
        batch_orders,
        sku_lookup,
        sku_zone_lookup,
        product_lookup,
        config,
        grid=grid,
        start=start,
        end=end,
        picker_speed_mps=picker_speed_mps,
    )
    return not batch_constraint_violations(summary, config)


def zone_limit_feasible(
    batch_orders: List[Order],
    candidate: Order,
    zone_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> bool:
    """Return True when adding *candidate* would not exceed the zone limit."""
    zones: Set[str] = set()
    for order in batch_orders:
        zones.update(order_zone_set(order, zone_lookup))
    zones.update(order_zone_set(candidate, zone_lookup))
    return len(zones) <= config.max_zones_per_batch


def category_limit_feasible(
    batch_orders: List[Order],
    candidate: Order,
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> bool:
    """Return True when candidate respects strict category grouping.
    
    If strict_category_grouping is enabled, a batch can only contain items
    that share the same category/categories. A candidate can only be added
    if its categories are a subset of the existing batch categories (or if
    the batch is empty).
    """
    if not getattr(config, "strict_category_grouping", False):
        return True

    batch_cats: Set[str] = set()
    for order in batch_orders:
        batch_cats.update(order_category_set(order, sku_category_lookup))
    
    if not batch_cats:
        return True
        
    cand_cats = order_category_set(candidate, sku_category_lookup)
    return cand_cats.issubset(batch_cats)

