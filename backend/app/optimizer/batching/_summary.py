"""_summary.py — BatchSummary computation with route estimation proxy.

This module centralises the logic for computing :class:`BatchSummary` objects,
which are consumed by constraint validators and scoring functions.  The route
estimator tries progressively cheaper proxies when full routing is unavailable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Set, Tuple

import networkx as nx

from app.optimizer.batching._types import BatchSummary
from app.optimizer.feature_engineering import (
    order_fragility_flags,
    order_pick_nodes,
    order_unit_count,
    order_volume,
    order_weight,
    order_zone_set,
)
from app.optimizer.graph_model import GridGraph
from app.optimizer.routing import route_distance, solve_route_nearest_neighbor
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Pick-target helpers
# ---------------------------------------------------------------------------

def batch_targets(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
) -> List[Coord]:
    """Return deduplicated pick targets for all orders in a batch."""
    nodes: List[Coord] = []
    for order in batch_orders:
        nodes.extend(order_pick_nodes(order, sku_lookup))
    return list(dict.fromkeys(nodes))


def reachable_batch_targets(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    grid: GridGraph,
) -> List[Coord]:
    """Return only those pick targets reachable from both entry and exit."""
    targets: List[Coord] = []
    for target in batch_targets(batch_orders, sku_lookup):
        if target not in grid.graph:
            continue
        if (
            nx.has_path(grid.graph, grid.entry, target)
            and nx.has_path(grid.graph, target, grid.exit)
        ):
            targets.append(target)
    return targets


# ---------------------------------------------------------------------------
# Route estimation
# ---------------------------------------------------------------------------

def _manhattan(a: Coord, b: Coord) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def estimate_route_distance_proxy(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    *,
    grid: GridGraph | None = None,
    start: Coord | None = None,
    end: Coord | None = None,
) -> float:
    """Estimate route distance without running the full TSP solver.

    Falls back gracefully through three levels of approximation:

    1. Grid-aware nearest-neighbour solve (when *grid*, *start*, and *end* are
       provided and all targets are reachable).
    2. Bounding-box heuristic using only the pick-target coordinates.
    3. Zero (when the batch has no pick targets).
    """
    raw_targets = batch_targets(batch_orders, sku_lookup)
    if not raw_targets:
        if grid is not None and start is not None and end is not None:
            try:
                return grid.travel_cost(start, end)
            except Exception:
                return 0.0
        return 0.0

    if grid is not None and start is not None and end is not None:
        filtered_targets = [t for t in raw_targets if t in grid.graph]
        if not filtered_targets:
            try:
                return grid.travel_cost(start, end)
            except Exception:
                return 0.0
        try:
            route = solve_route_nearest_neighbor(grid, start, end, filtered_targets)
            return route_distance(grid, route)
        except Exception:
            pass  # Fall through to bounding-box proxy

    # Bounding-box proxy
    xs = [c[0] for c in raw_targets]
    ys = [c[1] for c in raw_targets]
    bbox_span = float((max(xs) - min(xs)) + (max(ys) - min(ys)))
    _start: Coord = start if start is not None else raw_targets[0]
    _end: Coord = end if end is not None else raw_targets[0]
    start_cost = min(_manhattan(_start, t) for t in raw_targets)
    end_cost = min(_manhattan(t, _end) for t in raw_targets)
    return bbox_span + start_cost + end_cost + max(len(raw_targets) - 1, 0)


# ---------------------------------------------------------------------------
# BatchSummary builder
# ---------------------------------------------------------------------------

def summarize_batch(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    config: OptimizationConfig,
    *,
    grid: GridGraph | None = None,
    start: Coord | None = None,
    end: Coord | None = None,
    picker_speed_mps: float = 1.0,
) -> BatchSummary:
    """Build a :class:`BatchSummary` for *batch_orders*.

    All aggregate statistics (weight, volume, zones, fragility) are computed
    by iterating the orders once.  Route distance uses the proxy estimator.
    """
    zones: Set[str] = set()
    fragile = False
    bulky = False
    total_units = 0
    total_weight = 0.0
    total_volume = 0.0

    for order in batch_orders:
        zones.update(order_zone_set(order, sku_zone_lookup))
        total_units += order_unit_count(order)
        total_weight += order_weight(order, product_lookup)
        total_volume += order_volume(order, product_lookup)
        flags = order_fragility_flags(order, product_lookup)
        fragile = fragile or flags["fragile"]
        bulky = bulky or flags["bulky"]

    route_cost = estimate_route_distance_proxy(
        batch_orders, sku_lookup, grid=grid, start=start, end=end
    )
    duration_seconds = route_cost / max(picker_speed_mps, 0.1)

    return BatchSummary(
        order_count=len(batch_orders),
        total_units=total_units,
        total_weight=total_weight,
        total_volume=total_volume,
        target_count=len(batch_targets(batch_orders, sku_lookup)),
        route_distance=route_cost,
        duration_seconds=duration_seconds,
        zones=zones,
        fragile=fragile,
        bulky=bulky,
    )


def make_summary_cache(
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    config: OptimizationConfig,
    *,
    grid: GridGraph | None = None,
    start: Coord | None = None,
    end: Coord | None = None,
    picker_speed_mps: float = 1.0,
):
    """Return a memoised wrapper around :func:`summarize_batch`.

    The cache key is the sorted tuple of order IDs so that order is
    irrelevant and repeated evaluations of the same batch are free.
    """
    _cache: Dict[Tuple[str, ...], BatchSummary] = {}

    def cached(batch_orders: List[Order]) -> BatchSummary:
        key = tuple(sorted(o.order_id for o in batch_orders))
        if key in _cache:
            return _cache[key]
        result = summarize_batch(
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
        _cache[key] = result
        return result

    return cached
