"""batch_builder.py — Main batching algorithm drivers.

This module contains the five public batching strategies:
  - seed_distance_batching      (primary, zone/distance hybrid)
  - insertion_cost_batching     (insertion heuristic)
  - constrained_k_medoids       (spatial clustering)
  - greedy_capacity_batching    (fast greedy benchmark)
  - global_zone_task_batching   (territory-aware task decomposition)

All algorithmic logic is preserved exactly. Internal helpers have been
delegated to the batching subpackage modules.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Set, Tuple

import math
import numpy as np
import networkx as nx

from app.optimizer.batching.global_zone_task import (
    global_zone_task_batching,
)

from app.optimizer.batching._types import BatchAssignment, BatchSummary
from app.optimizer.batching._summary import (
    batch_targets,
    reachable_batch_targets,
    summarize_batch,
    make_summary_cache,
)
from app.optimizer.batching.assignment import (
    effective_batch_count,
    finalize_assignment,
    init_medoids,
    sorted_orders_for_assignment,
)
from app.optimizer.batching.constraints import (
    batch_constraint_violations,
    batch_feasible,
    category_limit_feasible,
)
from app.optimizer.batching.similarity import jaccard_similarity, order_category_set
from app.optimizer.feature_engineering import (
    order_centroid,
    order_fragility_flags,
    order_pick_nodes,
    order_unit_count,
    order_volume,
    order_weight,
    order_zone_set,
)
from app.optimizer.graph_model import GridGraph
from app.optimizer.routing import route_distance, solve_route_nearest_neighbor
from app.scoring.cost_function import order_cost, score_insertion
from app.schemas import OptimizationConfig, Order

Coord = Tuple[int, int]
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_neighbor_distance(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    grid: GridGraph,
    start: Coord,
    end: Coord,
) -> float:
    targets = reachable_batch_targets(batch_orders, sku_lookup, grid)
    route = solve_route_nearest_neighbor(grid, start, end, targets)
    return route_distance(grid, route)


def _spread_penalty(batch_targets_list: List[Coord], order_targets: List[Coord]) -> float:
    if not batch_targets_list or not order_targets:
        return 0.0
    center_x = sum(t[0] for t in batch_targets_list) / len(batch_targets_list)
    center_y = sum(t[1] for t in batch_targets_list) / len(batch_targets_list)
    return sum(abs(t[0] - center_x) + abs(t[1] - center_y) for t in order_targets) / len(order_targets)


def _new_aisle_count(batch_targets_list: List[Coord], order_targets: List[Coord]) -> int:
    existing_aisles = {t[0] for t in batch_targets_list}
    return len({t[0] for t in order_targets if t[0] not in existing_aisles})


def _seed_score(order: Order, sku_lookup: Dict[str, Coord], grid: GridGraph, depot: Coord) -> tuple:
    targets = reachable_batch_targets([order], sku_lookup, grid)
    distances: List[float] = []
    for target in targets:
        try:
            distances.append(grid.travel_cost(depot, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    farthest_distance = max(distances, default=0.0)
    return (order_unit_count(order), farthest_distance, -order.created_at_epoch)


def _dominant_zone(order: Order, sku_zone_lookup: Dict[str, str]) -> str | None:
    zones = [sku_zone_lookup.get(item.sku, "unknown") for item in order.items]
    if not zones:
        return None
    return Counter(zones).most_common(1)[0][0]


def _candidate_add_cost(
    batch_orders: List[Order],
    candidate: Order,
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
    grid: GridGraph,
    start: Coord,
    end: Coord,
    current_distance: float,
    sku_zone_lookup: Dict[str, str] | None = None,
) -> float:
    candidate_orders = batch_orders + [candidate]
    candidate_distance = _nearest_neighbor_distance(candidate_orders, sku_lookup, grid, start, end)
    route_delta = candidate_distance - current_distance
    batch_tgts = reachable_batch_targets(batch_orders, sku_lookup, grid)
    order_tgts = reachable_batch_targets([candidate], sku_lookup, grid)
    new_aisles = _new_aisle_count(batch_tgts, order_tgts)
    spread = _spread_penalty(batch_tgts, order_tgts)
    batch_cats = set().union(*[order_category_set(o, sku_category_lookup) for o in batch_orders])
    cand_cats = order_category_set(candidate, sku_category_lookup)
    cat_similarity = jaccard_similarity(batch_cats, cand_cats)
    zone_lookup = sku_zone_lookup or sku_category_lookup
    batch_zones = [z for o in batch_orders for z in order_zone_set(o, zone_lookup)]
    dominant_zone = Counter(batch_zones).most_common(1)[0][0] if batch_zones else None
    cand_zones = order_zone_set(candidate, zone_lookup)
    zone_penalty = 0.0
    if dominant_zone is not None and cand_zones != {dominant_zone}:
        zone_penalty = config.zone_mismatch_penalty_weight * len(cand_zones - {dominant_zone})
    same_zone_boost = (
        config.same_zone_nearby_boost_weight / max(1.0, spread)
        if dominant_zone is not None and cand_zones == {dominant_zone}
        else 0.0
    )
    return (
        route_delta
        + zone_penalty
        + config.advanced_aisle_penalty_weight * new_aisles
        + config.advanced_spread_penalty_weight * spread
        - config.advanced_category_boost_weight * cat_similarity
        - same_zone_boost
    )


def _merge_singletons(
    batches: List[List[Order]],
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    config: OptimizationConfig,
    grid: GridGraph,
    start: Coord,
    end: Coord,
    picker_speed_mps: float,
) -> None:
    if len(batches) <= 1:
        return
    idx = 0
    while idx < len(batches):
        batch = batches[idx]
        if len(batch) != 1 or len(batches) <= 1:
            idx += 1
            continue
        best_idx = -1
        best_delta = float("inf")
        for candidate_idx, candidate_batch in enumerate(batches):
            if candidate_idx == idx:
                continue
            merged = candidate_batch + batch
            merged_zones = {z for o in merged for z in order_zone_set(o, sku_zone_lookup)}
            if len(merged_zones) > config.max_zones_per_batch:
                continue
            if not batch_feasible(merged, sku_lookup, sku_zone_lookup, product_lookup, config, grid, start, end, picker_speed_mps):
                continue
            before = _nearest_neighbor_distance(candidate_batch, sku_lookup, grid, start, end)
            after = _nearest_neighbor_distance(merged, sku_lookup, grid, start, end)
            delta = after - before
            if delta < best_delta:
                best_delta = delta
                best_idx = candidate_idx
        if best_idx >= 0 and (
            config.advanced_singleton_merge_max_delta is None
            or best_delta <= config.advanced_singleton_merge_max_delta
        ):
            batches[best_idx].extend(batch)
            batches.pop(idx)
            continue
        idx += 1


def _merge_same_zone_batches(
    batches: List[List[Order]],
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    config: OptimizationConfig,
    grid: GridGraph,
    start: Coord,
    end: Coord,
    picker_speed_mps: float,
) -> None:
    changed = True
    while changed:
        changed = False
        for i in range(len(batches)):
            if changed:
                break
            zones_i = {z for o in batches[i] for z in order_zone_set(o, sku_zone_lookup)}
            if len(zones_i) != 1:
                continue
            for j in range(i + 1, len(batches)):
                zones_j = {z for o in batches[j] for z in order_zone_set(o, sku_zone_lookup)}
                if zones_i != zones_j:
                    continue
                merged = batches[i] + batches[j]
                if not batch_feasible(merged, sku_lookup, sku_zone_lookup, product_lookup, config, grid, start, end, picker_speed_mps):
                    continue
                before = (_nearest_neighbor_distance(batches[i], sku_lookup, grid, start, end)
                          + _nearest_neighbor_distance(batches[j], sku_lookup, grid, start, end))
                after = _nearest_neighbor_distance(merged, sku_lookup, grid, start, end)
                if after <= before:
                    batches[i] = merged
                    batches.pop(j)
                    changed = True
                    break


# ---------------------------------------------------------------------------
# Public algorithm: seed_distance_batching
# ---------------------------------------------------------------------------

def seed_distance_batching(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
    *,
    sku_zone_lookup: Dict[str, str] | None = None,
    product_lookup=None,
    grid: GridGraph,
    start: Coord,
    end: Coord,
    picker_speed_mps: float = 1.0,
) -> BatchAssignment:
    """Distance-first hybrid SDVRP algorithm (KMeans clustering + Capacity assignment)."""
    if not orders:
        return BatchAssignment(batches=[], labels={})

    zone_lookup = sku_zone_lookup or sku_category_lookup
    notes: List[str] = ["advanced_batching=distance_first_hybrid_sdvrp"]

    # STEP 1: Aggregation
    # We aggregate at the Order level in the backend to preserve the BatchAssignment schema.
    from sklearn.cluster import KMeans

    total_units = sum(order_unit_count(o) for o in orders)
    max_units = getattr(config, "max_batch_units", 25)
    k = math.ceil(total_units / max_units) if max_units > 0 else 1
    k = max(1, min(k, len(orders)))

    # STEP 2: KMeans Spatial Clustering
    coords = np.array([order_centroid(o, sku_lookup) for o in orders])
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    batches: List[List[Order]] = [[] for _ in range(k)]
    loads = [0] * k

    # STEP 3: Capacity-Aware Assignment
    for idx, order in enumerate(orders):
        cluster = labels[idx]
        qty = order_unit_count(order)

        best_cluster = cluster
        if loads[cluster] + qty > max_units:
            # nearest cluster with space
            best_dist = float("inf")
            order_coord = coords[idx]
            for ci in range(k):
                if loads[ci] + qty <= max_units:
                    center = kmeans.cluster_centers_[ci]
                    d = abs(order_coord[0] - center[0]) + abs(order_coord[1] - center[1])
                    if d < best_dist:
                        best_dist = d
                        best_cluster = ci
                        
        batches[best_cluster].append(order)
        loads[best_cluster] += qty
        
    # Remove empty batches
    batches = [b for b in batches if b]
    k = len(batches)

    # STEP 4: Lightweight Congestion Repair (Local Swap)
    for _ in range(2): # Lightweight 2 iterations
        for i in range(k):
            for j in range(i+1, k):
                if not batches[i] or not batches[j]:
                    continue
                
                ri = [order_centroid(o, sku_lookup) for o in batches[i]]
                rj = [order_centroid(o, sku_lookup) for o in batches[j]]
                
                # Check bounding box overlap as a fast proxy for route intersection
                if not ri or not rj: continue
                min_xi, max_xi = min(c[0] for c in ri), max(c[0] for c in ri)
                min_yi, max_yi = min(c[1] for c in ri), max(c[1] for c in ri)
                min_xj, max_xj = min(c[0] for c in rj), max(c[0] for c in rj)
                min_yj, max_yj = min(c[1] for c in rj), max(c[1] for c in rj)
                
                overlap = not (max_xi < min_xj or min_xi > max_xj or max_yi < min_yj or min_yi > max_yj)
                
                if not overlap:
                    continue
                    
                # Swap farthest orders
                ci_order = max(batches[i], key=lambda o: abs(start[0] - order_centroid(o, sku_lookup)[0]) + abs(start[1] - order_centroid(o, sku_lookup)[1]))
                cj_order = max(batches[j], key=lambda o: abs(start[0] - order_centroid(o, sku_lookup)[0]) + abs(start[1] - order_centroid(o, sku_lookup)[1]))
                
                # Ensure capacity constraints
                if loads[i] - order_unit_count(ci_order) + order_unit_count(cj_order) > max_units: continue
                if loads[j] - order_unit_count(cj_order) + order_unit_count(ci_order) > max_units: continue
                
                # Check pure distance improvement
                before_dist = _nearest_neighbor_distance(batches[i], sku_lookup, grid, start, end) + \
                              _nearest_neighbor_distance(batches[j], sku_lookup, grid, start, end)
                              
                temp_i = [cj_order if o == ci_order else o for o in batches[i]]
                temp_j = [ci_order if o == cj_order else o for o in batches[j]]
                
                after_dist = _nearest_neighbor_distance(temp_i, sku_lookup, grid, start, end) + \
                             _nearest_neighbor_distance(temp_j, sku_lookup, grid, start, end)
                             
                if after_dist < before_dist:
                    batches[i] = temp_i
                    batches[j] = temp_j
                    loads[i] = loads[i] - order_unit_count(ci_order) + order_unit_count(cj_order)
                    loads[j] = loads[j] - order_unit_count(cj_order) + order_unit_count(ci_order)
                    notes.append(f"lightweight_swap optimized distance: {before_dist:.1f} -> {after_dist:.1f}")

    # Finalization
    batch_names = [f"{config.batch_id_prefix}-{i}" for i in range(len(batches))]
    batch_types = ["standard"] * len(batches)
    
    for batch_name, batch in zip(batch_names, batches):
        distance = _nearest_neighbor_distance(batch, sku_lookup, grid, start, end)
        total_items = sum(order_unit_count(o) for o in batch)
        notes.append(f"Batch ID={batch_name} Total items={total_items} Total distance={distance:.2f}")
    
    return finalize_assignment(batches, batch_names, batch_types, notes=notes)


# ---------------------------------------------------------------------------
# Public algorithm: insertion_cost_batching
# ---------------------------------------------------------------------------

def insertion_cost_batching(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
    *,
    sku_zone_lookup: Dict[str, str] | None = None,
    product_lookup=None,
    grid: GridGraph | None = None,
    start: Coord | None = None,
    end: Coord | None = None,
    picker_speed_mps: float = 1.0,
) -> BatchAssignment:
    """Marginal-route-cost insertion batching."""
    if not orders:
        return BatchAssignment(batches=[], labels={})

    zone_lookup = sku_zone_lookup or sku_category_lookup
    target_standard_batches = effective_batch_count(orders, sku_lookup, sku_category_lookup, config)
    hard_standard_batch_cap = (
        target_standard_batches
        if not config.dynamic_batching_enabled
        else min(config.employee_count, len(orders))
    )
    batches: List[List[Order]] = []
    batch_names: List[str] = []
    batch_types: List[str] = []
    notes: List[str] = []
    exception_order_ids: List[str] = []
    overflow_counter = 0
    extra_standard_batches = 0

    summarize_cached = make_summary_cache(
        sku_lookup, zone_lookup, product_lookup, config,
        grid=grid, start=start, end=end, picker_speed_mps=picker_speed_mps,
    )

    for order in sorted_orders_for_assignment(orders):
        single_summary = summarize_cached([order])
        standalone_violations = batch_constraint_violations(single_summary, config)
        if standalone_violations:
            notes.append(f"order {order.order_id} infeasible as standalone batch: {', '.join(standalone_violations)}")
            exception_order_ids.append(order.order_id)
            continue

        best_idx = -1
        best_score = float("inf")
        best_category_similarity = 0.0
        best_zone_similarity = 0.0
        order_zones = order_zone_set(order, zone_lookup)
        order_categories = order_category_set(order, sku_category_lookup)
        order_flags = order_fragility_flags(order, product_lookup)

        for idx, batch in enumerate(batches):
            before_summary = summarize_cached(batch)
            candidate_batch = batch + [order]
            after_summary = summarize_cached(candidate_batch)
            if batch_constraint_violations(after_summary, config):
                continue
            if not category_limit_feasible(batch, order, sku_category_lookup, config):
                continue

            batch_sku_set = {item.sku for grouped_order in batch for item in grouped_order.items}
            category_similarity = jaccard_similarity(
                order_categories,
                {sku_category_lookup.get(sku, sku.split("-", 1)[0]) for sku in batch_sku_set},
            )
            zone_similarity = (
                jaccard_similarity(order_zones, before_summary.zones)
                if before_summary.zones else 0.0
            )
            route_delta = after_summary.route_distance - before_summary.route_distance

            breakdown = score_insertion(
                order, batch, before_summary, after_summary, route_delta,
                order_zones, order_categories, order_flags, batch_sku_set,
                sku_category_lookup, batch_types[idx], config,
            )
            score = breakdown.total
            if score < best_score:
                best_score = score
                best_idx = idx
                best_category_similarity = category_similarity
                best_zone_similarity = zone_similarity

        standard_batches_in_use = sum(1 for bt in batch_types if bt == "standard")
        should_seed_new_standard = (
            best_idx >= 0
            and standard_batches_in_use < target_standard_batches
            and batch_types[best_idx] == "standard"
            and max(best_category_similarity, best_zone_similarity) < 0.25
        )

        if best_idx >= 0 and not should_seed_new_standard:
            batches[best_idx].append(order)
            continue

        if standard_batches_in_use < target_standard_batches:
            batches.append([order])
            batch_names.append(f"{config.batch_id_prefix}-{standard_batches_in_use}")
            batch_types.append("standard")
            continue

        if standard_batches_in_use < hard_standard_batch_cap:
            batches.append([order])
            batch_names.append(f"{config.batch_id_prefix}-{standard_batches_in_use}")
            batch_types.append("standard")
            extra_standard_batches += 1
            continue

        if config.allow_overflow_batches and (
            not config.dynamic_batching_enabled or len(batches) < config.employee_count
        ):
            overflow_name = f"{config.overflow_batch_name_prefix}-{overflow_counter}"
            overflow_counter += 1
            batches.append([order])
            batch_names.append(overflow_name)
            batch_types.append("overflow")
            notes.append(f"order {order.order_id} assigned to overflow batch {overflow_name}")
            continue

        notes.append(f"order {order.order_id} could not be assigned without violating hard constraints")
        exception_order_ids.append(order.order_id)

    if extra_standard_batches > 0:
        notes.append(f"opened {extra_standard_batches} extra standard batches beyond target to preserve hard constraints")

    return finalize_assignment(batches, batch_names, batch_types, notes=notes, exception_order_ids=exception_order_ids)


# ---------------------------------------------------------------------------
# Public algorithm: constrained_k_medoids
# ---------------------------------------------------------------------------

def constrained_k_medoids(
    orders: List[Order],
    sku_lookup: Dict[str, Tuple[int, int]],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> BatchAssignment:
    """Capacity-constrained k-medoids spatial clustering."""
    if not orders:
        return BatchAssignment(batches=[], labels={})

    k = effective_batch_count(orders, sku_lookup, sku_category_lookup, config)
    features = np.array([order_centroid(o, sku_lookup) for o in orders], dtype=np.float64)
    medoid_indices = init_medoids(features, k)
    medoids = features[medoid_indices]
    max_iter = 12
    labels = np.zeros(len(orders), dtype=np.int32)
    order_category_sets = [order_category_set(o, sku_category_lookup) for o in orders]
    overflow_orders: List[Order] = []
    exception_order_ids: List[str] = []
    notes: List[str] = []

    for _ in range(max_iter):
        batch_weights = [0.0] * k
        batch_sizes = [0] * k
        batch_units = [0] * k
        batch_category_sets = [set() for _ in range(k)]
        new_labels = np.full(len(orders), -1, dtype=np.int32)
        overflow_orders = []
        exception_order_ids = []

        order_indices = sorted(range(len(orders)), key=lambda i: orders[i].due_time_minutes)
        for i in order_indices:
            o = orders[i]
            weight = order_weight(o)
            costs = []
            for batch_idx in range(k):
                cost = order_cost(
                    features[i], medoids[batch_idx], o.due_time_minutes,
                    weight, order_category_sets[i], batch_category_sets[batch_idx], config,
                )
                feasible = (
                    batch_sizes[batch_idx] < config.max_batch_size
                    and batch_weights[batch_idx] + weight <= config.max_batch_weight
                    and batch_units[batch_idx] + order_unit_count(o) <= getattr(config, "max_batch_units", 25)
                )
                if getattr(config, "strict_category_grouping", False) and batch_category_sets[batch_idx]:
                    if not order_category_sets[i].issubset(batch_category_sets[batch_idx]):
                        feasible = False
                costs.append((cost if feasible else float("inf"), batch_idx))
            costs.sort(key=lambda x: x[0])
            if costs and costs[0][0] < float("inf"):
                chosen = costs[0][1]
                new_labels[i] = chosen
                batch_sizes[chosen] += 1
                batch_weights[chosen] += weight
                batch_units[chosen] += order_unit_count(o)
                batch_category_sets[chosen].update(order_category_sets[i])
                continue
            if config.allow_overflow_batches and weight <= config.max_batch_weight:
                overflow_orders.append(o)
            else:
                exception_order_ids.append(o.order_id)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for batch_idx in range(k):
            member_idx = np.where(labels == batch_idx)[0]
            if len(member_idx) == 0:
                continue
            member_features = features[member_idx]
            pairwise = np.sum(
                np.linalg.norm(member_features[:, None, :] - member_features[None, :, :], axis=2),
                axis=1,
            )
            medoids[batch_idx] = member_features[int(np.argmin(pairwise))]

    standard_batches: List[List[Order]] = [[] for _ in range(k)]
    for idx, o in enumerate(orders):
        batch_idx = int(labels[idx])
        if batch_idx >= 0:
            standard_batches[batch_idx].append(o)

    overflow_batches = [[o] for o in overflow_orders]
    if overflow_batches:
        notes.append("legacy k-medoids placed infeasible leftovers into overflow singleton batches")
    if exception_order_ids:
        notes.append("legacy k-medoids could not assign some orders without violating hard constraints")

    batch_names = [f"{config.batch_id_prefix}-{i}" for i in range(len(standard_batches))] + [
        f"{config.overflow_batch_name_prefix}-{i}" for i in range(len(overflow_batches))
    ]
    batch_types = ["standard"] * len(standard_batches) + ["overflow"] * len(overflow_batches)
    return finalize_assignment(
        standard_batches + overflow_batches, batch_names, batch_types,
        notes=notes, exception_order_ids=exception_order_ids,
    )


# ---------------------------------------------------------------------------
# Public algorithm: greedy_capacity_batching
# ---------------------------------------------------------------------------

def greedy_capacity_batching(
    orders: List[Order],
    sku_lookup: Dict[str, Tuple[int, int]],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> BatchAssignment:
    """Fast greedy batching sorted by due time (benchmark strategy)."""
    if not orders:
        return BatchAssignment(batches=[], labels={})

    sorted_orders = sorted(orders, key=lambda o: o.due_time_minutes)
    batches: List[List[Order]] = []
    batch_weights: List[float] = []
    batch_units: List[int] = []
    max_batches = effective_batch_count(orders, sku_lookup, sku_category_lookup, config)
    notes: List[str] = []
    exception_order_ids: List[str] = []
    overflow_batches: List[List[Order]] = []

    for order in sorted_orders:
        weight = order_weight(order)
        if weight > config.max_batch_weight:
            exception_order_ids.append(order.order_id)
            notes.append(f"order {order.order_id} exceeds max_batch_weight as a standalone batch")
            continue

        order_cent = order_centroid(order, sku_lookup)
        best_idx = -1
        best_cost = float("inf")

        for idx, group in enumerate(batches):
            if len(group) >= config.max_batch_size:
                continue
            if batch_weights[idx] + weight > config.max_batch_weight:
                continue
            if batch_units[idx] + order_unit_count(order) > getattr(config, "max_batch_units", 25):
                continue
            if not category_limit_feasible(group, order, sku_category_lookup, config):
                continue
            group_centroid = np.mean([order_centroid(go, sku_lookup) for go in group], axis=0)
            cost = float(np.linalg.norm(order_cent - group_centroid))
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        if best_idx == -1 and len(batches) < max_batches:
            batches.append([order])
            batch_weights.append(weight)
            batch_units.append(order_unit_count(order))
            continue

        if best_idx == -1:
            if config.allow_overflow_batches:
                overflow_batches.append([order])
                notes.append(f"order {order.order_id} sent to overflow in greedy benchmark mode")
            else:
                exception_order_ids.append(order.order_id)
            continue

        batches[best_idx].append(order)
        batch_weights[best_idx] += weight
        batch_units[best_idx] += order_unit_count(order)

    batch_names = [f"{config.batch_id_prefix}-{i}" for i in range(len(batches))] + [
        f"{config.overflow_batch_name_prefix}-{i}" for i in range(len(overflow_batches))
    ]
    batch_types = ["standard"] * len(batches) + ["overflow"] * len(overflow_batches)
    return finalize_assignment(
        batches + overflow_batches, batch_names, batch_types,
        notes=notes, exception_order_ids=exception_order_ids,
    )
