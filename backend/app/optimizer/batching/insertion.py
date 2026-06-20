"""insertion.py — Route insertion helpers for item-level batching.

These functions handle the geometry of inserting items into an existing route,
computing marginal route costs, and flattening order items to warehouse
coordinates.  They are consumed by both the batch_builder algorithms and the
DHOBR item-level scoring.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from app.optimizer.batching._types import BatchItem
from app.optimizer.graph_model import GridGraph
from app.optimizer.routing import route_distance
from app.schemas import Order

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Route geometry
# ---------------------------------------------------------------------------

def route_node_distance(grid: GridGraph, route: List[Coord]) -> float:
    """Return the total travel cost of *route* through *grid*."""
    return route_distance(grid, route) if len(route) > 1 else 0.0


def insert_into_route(
    grid: GridGraph,
    route: List[Coord],
    item_coord: Coord,
) -> List[Coord]:
    """Insert *item_coord* into *route* at the cheapest position.

    The cheapest position minimises the marginal cost
    ``travel(prev, item) + travel(item, next) - travel(prev, next)``.
    """
    if item_coord in route:
        return list(route)
    if len(route) < 2:
        return list(route) + [item_coord]

    best_pos = 1
    best_delta = float("inf")
    for pos in range(1, len(route)):
        prev_node = route[pos - 1]
        next_node = route[pos]
        delta = (
            grid.travel_cost(prev_node, item_coord)
            + grid.travel_cost(item_coord, next_node)
            - grid.travel_cost(prev_node, next_node)
        )
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    updated = list(route)
    updated.insert(best_pos, item_coord)
    return updated


def simulate_insertion(
    grid: GridGraph,
    route: List[Coord],
    items: List[BatchItem],
) -> tuple[List[Coord], float]:
    """Simulate inserting all *items* into *route*.

    Returns the updated route and the marginal route-distance increase.
    """
    temp_route = list(route)
    before = route_node_distance(grid, temp_route)
    for item in items:
        temp_route = insert_into_route(grid, temp_route, item.coord)
    return temp_route, route_node_distance(grid, temp_route) - before


# ---------------------------------------------------------------------------
# Order item flattening
# ---------------------------------------------------------------------------

def flatten_order_items(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    sku_zone_lookup: Dict[str, str],
    grid: GridGraph,
) -> Dict[str, List[BatchItem]]:
    """Resolve each order's items to warehouse coordinates.

    Items whose SKU is unmapped, outside the walkable graph, or not reachable
    from both entry and exit are silently dropped.

    Returns a mapping of ``order_id → [BatchItem, ...]``.
    """
    flattened: Dict[str, List[BatchItem]] = {}
    for order in orders:
        entries: List[BatchItem] = []
        for item in order.items:
            coord = sku_lookup.get(item.sku)
            if coord is None or coord not in grid.graph:
                continue
            if not nx.has_path(grid.graph, grid.entry, coord) or not nx.has_path(
                grid.graph, coord, grid.exit
            ):
                continue
            entries.append(
                BatchItem(
                    order_id=order.order_id,
                    sku=item.sku,
                    qty=item.qty,
                    coord=coord,
                    category=sku_category_lookup.get(
                        item.sku, item.sku.split("-", 1)[0]
                    ),
                    zone=sku_zone_lookup.get(
                        item.sku,
                        sku_category_lookup.get(item.sku, item.sku.split("-", 1)[0]),
                    ),
                    due_time_minutes=order.due_time_minutes,
                    created_at_epoch=order.created_at_epoch,
                )
            )
        flattened[order.order_id] = entries
    return flattened
