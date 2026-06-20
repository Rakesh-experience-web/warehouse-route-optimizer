from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx

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
from app.schemas import OptimizationConfig, Order


Coord = Tuple[int, int]


@dataclass
class BatchAssignment:
    batches: List[List[Order]]
    labels: Dict[str, int]
    batch_names: List[str] = field(default_factory=list)
    batch_types: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    exception_order_ids: List[str] = field(default_factory=list)
    batch_items: List[List["BatchItem"]] = field(default_factory=list)
    picker_ids: List[str] = field(default_factory=list)


@dataclass
class BatchSummary:
    order_count: int
    total_units: int
    total_weight: float
    total_volume: float
    target_count: int
    route_distance: float
    duration_seconds: float
    zones: Set[str]
    fragile: bool
    bulky: bool


@dataclass(frozen=True)
class BatchItem:
    order_id: str
    sku: str
    qty: int
    coord: Coord
    category: str
    zone: str
    due_time_minutes: int
    created_at_epoch: int


@dataclass
class SmartBatch:
    batch_id: str
    items: List[BatchItem]
    route: List[Coord]
    picker_id: str
    current_load: int = 0


def _average_pairwise_similarity(order_sku_sets: List[Set[str]]) -> float:
    n = len(order_sku_sets)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _jaccard_similarity(order_sku_sets[i], order_sku_sets[j])
            pairs += 1
    return total / pairs if pairs > 0 else 1.0


def _order_category_set(order: Order, sku_category_lookup: Dict[str, str]) -> Set[str]:
    return {sku_category_lookup.get(item.sku, item.sku.split("-", 1)[0]) for item in order.items}


def _effective_batch_count(
    orders: List[Order],
    sku_lookup: Dict[str, Tuple[int, int]],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> int:
    if not orders:
        return 0
    min_batches_by_size = max(1, math.ceil(len(orders) / max(config.max_batch_size, 1)))
    total_weight = sum(order_weight(order) for order in orders)
    min_batches_by_weight = max(1, math.ceil(total_weight / max(config.max_batch_weight, config.min_capacity_denominator)))
    capacity_floor = max(min_batches_by_size, min_batches_by_weight)
    if not config.dynamic_batching_enabled:
        return min(max(config.batch_count, 1), len(orders))

    desired_upper = min(max(config.batch_count, capacity_floor, 1), config.employee_count, len(orders))
    if desired_upper == 1:
        return 1

    order_category_sets: List[Set[str]] = [_order_category_set(o, sku_category_lookup) for o in orders]
    avg_similarity = _average_pairwise_similarity(order_category_sets)
    dissimilarity = max(0.0, 1.0 - avg_similarity)
    # Dynamic batching should adapt the total number of planned batches, not
    # cap it to simultaneous pickers. Higher order volumes still need multiple
    # execution waves even when only a few pickers are active concurrently.
    similarity_target = 1 + int(math.floor(dissimilarity * (desired_upper - 1)))
    k = max(capacity_floor, similarity_target)
    return min(max(k, 1), config.employee_count, len(orders))


def _init_medoids(features: np.ndarray, k: int) -> List[int]:
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


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _order_cost(
    feature: np.ndarray,
    medoid: np.ndarray,
    due_time_minutes: int,
    weight: float,
    order_categories: Set[str],
    batch_categories: Set[str],
    config: OptimizationConfig,
) -> float:
    distance_cost = np.linalg.norm(feature - medoid)
    due_risk = 1.0 / max(due_time_minutes, 1)
    similarity_bonus = _jaccard_similarity(order_categories, batch_categories) if batch_categories else 0.0
    return (
        config.alpha_distance * distance_cost
        + config.beta_due_time * due_risk
        + config.gamma_weight * weight
        - config.delta_similarity * similarity_bonus
    )


def _sorted_orders_for_assignment(orders: List[Order]) -> List[Order]:
    def key(order: Order) -> tuple[float, int, float, int]:
        urgency_floor = order.latest_pick_start_minutes if order.latest_pick_start_minutes is not None else order.due_time_minutes
        priority = float(order.priority or 0.0)
        return (float(urgency_floor), order.due_time_minutes, -priority, order.created_at_epoch)

    return sorted(orders, key=key)


def _batch_targets(batch_orders: List[Order], sku_lookup: Dict[str, Coord]) -> List[Coord]:
    nodes: List[Coord] = []
    for order in batch_orders:
        nodes.extend(order_pick_nodes(order, sku_lookup))
    return list(dict.fromkeys(nodes))


def _manhattan(a: Coord, b: Coord) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def _estimate_route_distance_proxy(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    *,
    grid: GridGraph | None = None,
    start: Coord | None = None,
    end: Coord | None = None,
) -> float:
    raw_targets = _batch_targets(batch_orders, sku_lookup)
    if not raw_targets:
        if grid is not None and start is not None and end is not None:
            try:
                return grid.travel_cost(start, end)
            except Exception:
                return 0.0
        return 0.0

    if grid is not None and start is not None and end is not None:
        filtered_targets = [target for target in raw_targets if target in grid.graph]
        if not filtered_targets:
            try:
                return grid.travel_cost(start, end)
            except Exception:
                return 0.0
        try:
            route = solve_route_nearest_neighbor(grid, start, end, filtered_targets)
            return route_distance(grid, route)
        except Exception:
            pass

    xs = [coord[0] for coord in raw_targets]
    ys = [coord[1] for coord in raw_targets]
    bbox_span = float((max(xs) - min(xs)) + (max(ys) - min(ys)))
    if start is None:
        start = raw_targets[0]
    if end is None:
        end = raw_targets[0]
    start_cost = min(_manhattan(start, target) for target in raw_targets)
    end_cost = min(_manhattan(target, end) for target in raw_targets)
    return bbox_span + start_cost + end_cost + max(len(raw_targets) - 1, 0)


def _summarize_batch(
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

    route_cost = _estimate_route_distance_proxy(batch_orders, sku_lookup, grid=grid, start=start, end=end)
    duration_seconds = route_cost / max(picker_speed_mps, 0.1)
    return BatchSummary(
        order_count=len(batch_orders),
        total_units=total_units,
        total_weight=total_weight,
        total_volume=total_volume,
        target_count=len(_batch_targets(batch_orders, sku_lookup)),
        route_distance=route_cost,
        duration_seconds=duration_seconds,
        zones=zones,
        fragile=fragile,
        bulky=bulky,
    )


def _batch_constraint_violations(summary: BatchSummary, config: OptimizationConfig) -> List[str]:
    violations: List[str] = []
    if summary.order_count > config.max_batch_size:
        violations.append("max_batch_size")
    if summary.total_weight > config.max_batch_weight:
        violations.append("max_batch_weight")
    if config.max_batch_volume is not None and summary.total_volume > config.max_batch_volume:
        violations.append("max_batch_volume")
    if summary.target_count > config.max_shelf_visits_per_picker:
        violations.append("max_shelf_visits_per_picker")
    if config.max_batch_duration_seconds is not None and summary.duration_seconds > config.max_batch_duration_seconds:
        violations.append("max_batch_duration_seconds")
    return violations


def _finalize_assignment(
    batches: List[List[Order]],
    batch_names: List[str],
    batch_types: List[str],
    *,
    notes: List[str] | None = None,
    exception_order_ids: List[str] | None = None,
) -> BatchAssignment:
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
    if not orders:
        return BatchAssignment(batches=[], labels={})

    zone_lookup = sku_zone_lookup or sku_category_lookup
    target_standard_batches = _effective_batch_count(orders, sku_lookup, sku_category_lookup, config)
    hard_standard_batch_cap = target_standard_batches if not config.dynamic_batching_enabled else min(config.employee_count, len(orders))
    batches: List[List[Order]] = []
    batch_names: List[str] = []
    batch_types: List[str] = []
    notes: List[str] = []
    exception_order_ids: List[str] = []
    overflow_counter = 0
    extra_standard_batches = 0
    summary_cache: Dict[Tuple[str, ...], BatchSummary] = {}

    def summarize_batch_cached(batch_orders: List[Order]) -> BatchSummary:
        batch_key = tuple(sorted(order.order_id for order in batch_orders))
        cached = summary_cache.get(batch_key)
        if cached is not None:
            return cached
        summary = _summarize_batch(
            batch_orders,
            sku_lookup,
            zone_lookup,
            product_lookup,
            config,
            grid=grid,
            start=start,
            end=end,
            picker_speed_mps=picker_speed_mps,
        )
        summary_cache[batch_key] = summary
        return summary

    # This replaces the centroid-only prototype behavior with a marginal route
    # insertion proxy, while still using cheap route estimates to stay fast.
    for order in _sorted_orders_for_assignment(orders):
        single_summary = summarize_batch_cached([order])
        standalone_violations = _batch_constraint_violations(single_summary, config)
        if standalone_violations:
            notes.append(
                f"order {order.order_id} infeasible as standalone batch: {', '.join(standalone_violations)}"
            )
            exception_order_ids.append(order.order_id)
            continue

        best_idx = -1
        best_score = float("inf")
        best_category_similarity = 0.0
        best_zone_similarity = 0.0
        order_zones = order_zone_set(order, zone_lookup)
        order_categories = _order_category_set(order, sku_category_lookup)
        order_flags = order_fragility_flags(order, product_lookup)

        for idx, batch in enumerate(batches):
            before_summary = summarize_batch_cached(batch)
            candidate_batch = batch + [order]
            after_summary = summarize_batch_cached(candidate_batch)
            if _batch_constraint_violations(after_summary, config):
                continue

            batch_categories = {item.sku for grouped_order in batch for item in grouped_order.items}
            category_similarity = _jaccard_similarity(
                order_categories,
                {sku_category_lookup.get(sku, sku.split("-", 1)[0]) for sku in batch_categories},
            )
            zone_similarity = _jaccard_similarity(order_zones, before_summary.zones) if before_summary.zones else 0.0
            route_delta = after_summary.route_distance - before_summary.route_distance
            urgency_window = (
                order.latest_pick_start_minutes
                if order.latest_pick_start_minutes is not None
                else float(order.due_time_minutes)
            )
            urgency_penalty = max(0.0, (after_summary.duration_seconds / 60.0) - urgency_window)
            workload_penalty = before_summary.order_count / max(config.max_batch_size, 1)
            workload_penalty += before_summary.total_weight / max(config.max_batch_weight, config.min_capacity_denominator)
            if config.max_batch_volume is not None and config.max_batch_volume > 0:
                workload_penalty += before_summary.total_volume / config.max_batch_volume
            fragility_penalty = 0.0
            if order_flags["fragile"] and before_summary.bulky:
                fragility_penalty += config.fragile_bulky_penalty
            if order_flags["bulky"] and before_summary.fragile:
                fragility_penalty += config.fragile_bulky_penalty
            if order.temperature_sensitive and before_summary.zones and "ambient" in before_summary.zones and order_zones:
                fragility_penalty += config.temperature_zone_mismatch_penalty
            priority_bonus = float(order.priority or 0.0) * config.priority_score_weight
            overflow_penalty = config.overflow_assignment_penalty if batch_types[idx] == "overflow" else 0.0
            score = (
                config.route_cost_reweight_factor * route_delta
                + config.beta_due_time * urgency_penalty
                + config.gamma_weight * workload_penalty
                + config.delta_similarity * (1.0 - zone_similarity)
                + fragility_penalty
                + overflow_penalty
                - config.similarity_batch_boost * category_similarity
                - priority_bonus
            )
            if score < best_score:
                best_score = score
                best_idx = idx
                best_category_similarity = category_similarity
                best_zone_similarity = zone_similarity

        standard_batches_in_use = sum(1 for batch_type in batch_types if batch_type == "standard")
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

        notes.append("order %s could not be assigned without violating hard constraints" % order.order_id)
        exception_order_ids.append(order.order_id)

        notes.append("order %s could not be assigned without violating hard constraints" % order.order_id)
        exception_order_ids.append(order.order_id)

    if extra_standard_batches > 0:
        notes.append(f"opened {extra_standard_batches} extra standard batches beyond target to preserve hard constraints")

    return _finalize_assignment(
        batches,
        batch_names,
        batch_types,
        notes=notes,
        exception_order_ids=exception_order_ids,
    )


def _reachable_batch_targets(batch_orders: List[Order], sku_lookup: Dict[str, Coord], grid: GridGraph) -> List[Coord]:
    targets: List[Coord] = []
    for target in _batch_targets(batch_orders, sku_lookup):
        if target not in grid.graph:
            continue
        if nx.has_path(grid.graph, grid.entry, target) and nx.has_path(grid.graph, target, grid.exit):
            targets.append(target)
    return targets


def _nearest_neighbor_distance(
    batch_orders: List[Order],
    sku_lookup: Dict[str, Coord],
    grid: GridGraph,
    start: Coord,
    end: Coord,
) -> float:
    targets = _reachable_batch_targets(batch_orders, sku_lookup, grid)
    route = solve_route_nearest_neighbor(grid, start, end, targets)
    return route_distance(grid, route)


def _route_node_distance(grid: GridGraph, route: List[Coord]) -> float:
    return route_distance(grid, route) if len(route) > 1 else 0.0


def _insert_into_route(grid: GridGraph, route: List[Coord], item_coord: Coord) -> List[Coord]:
    if item_coord in route:
        return list(route)
    if len(route) < 2:
        return list(route) + [item_coord]

    best_pos = 1
    best_delta = float("inf")
    for pos in range(1, len(route)):
        prev_node = route[pos - 1]
        next_node = route[pos]
        delta = grid.travel_cost(prev_node, item_coord) + grid.travel_cost(item_coord, next_node) - grid.travel_cost(prev_node, next_node)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    updated = list(route)
    updated.insert(best_pos, item_coord)
    return updated


def _simulate_insertion(grid: GridGraph, route: List[Coord], items: List[BatchItem]) -> tuple[List[Coord], float]:
    temp_route = list(route)
    before = _route_node_distance(grid, temp_route)
    for item in items:
        temp_route = _insert_into_route(grid, temp_route, item.coord)
    return temp_route, _route_node_distance(grid, temp_route) - before


def _flatten_order_items(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    sku_zone_lookup: Dict[str, str],
    grid: GridGraph,
) -> Dict[str, List[BatchItem]]:
    flattened: Dict[str, List[BatchItem]] = {}
    for order in orders:
        entries: List[BatchItem] = []
        for item in order.items:
            coord = sku_lookup.get(item.sku)
            if coord is None or coord not in grid.graph:
                continue
            if not nx.has_path(grid.graph, grid.entry, coord) or not nx.has_path(grid.graph, coord, grid.exit):
                continue
            entries.append(
                BatchItem(
                    order_id=order.order_id,
                    sku=item.sku,
                    qty=item.qty,
                    coord=coord,
                    category=sku_category_lookup.get(item.sku, item.sku.split("-", 1)[0]),
                    zone=sku_zone_lookup.get(item.sku, sku_category_lookup.get(item.sku, item.sku.split("-", 1)[0])),
                    due_time_minutes=order.due_time_minutes,
                    created_at_epoch=order.created_at_epoch,
                )
            )
        flattened[order.order_id] = entries
    return flattened


def _item_similarity_penalty(grid: GridGraph, batch: SmartBatch, items: List[BatchItem]) -> float:
    if not batch.items or not items:
        return 0.0
    distances: List[float] = []
    for item in items:
        distances.extend(grid.travel_cost(item.coord, existing.coord) for existing in batch.items)
    return sum(distances) / len(distances) if distances else 0.0


def _picker_load_penalty(batch: SmartBatch, config: OptimizationConfig) -> float:
    soft_capacity = max(config.max_batch_size, 1)
    return batch.current_load / soft_capacity


def _delay_penalty(route_distance_after: float, items: List[BatchItem], picker_speed_mps: float) -> float:
    if not items:
        return 0.0
    estimated_finish_minutes = (route_distance_after / max(picker_speed_mps, 0.1)) / 60.0
    due_time = min(item.due_time_minutes for item in items)
    return max(0.0, estimated_finish_minutes - due_time)


def _category_boost(batch: SmartBatch, items: List[BatchItem], config: OptimizationConfig) -> float:
    if not batch.items or not items:
        return 0.0
    batch_categories = {item.category for item in batch.items}
    item_categories = {item.category for item in items}
    return config.advanced_category_boost_weight * _jaccard_similarity(batch_categories, item_categories)


def _dominant_zone_from_items(items: List[BatchItem]) -> str | None:
    if not items:
        return None
    return Counter(item.zone for item in items).most_common(1)[0][0]


def _zone_set(items: List[BatchItem]) -> Set[str]:
    return {item.zone for item in items}


def _zone_mismatch_penalty(batch: SmartBatch, items: List[BatchItem], config: OptimizationConfig) -> float:
    dominant_zone = _dominant_zone_from_items(batch.items)
    if dominant_zone is None:
        return 0.0
    incoming_zones = _zone_set(items)
    if incoming_zones == {dominant_zone}:
        return 0.0
    return config.zone_mismatch_penalty_weight * len(incoming_zones - {dominant_zone})


def _would_exceed_zone_limit(batch: SmartBatch, items: List[BatchItem], config: OptimizationConfig) -> bool:
    return len(_zone_set(batch.items + items)) > config.max_zones_per_batch


def _same_zone_nearby_boost(grid: GridGraph, batch: SmartBatch, items: List[BatchItem], config: OptimizationConfig) -> float:
    dominant_zone = _dominant_zone_from_items(batch.items)
    if dominant_zone is None or not items or any(item.zone != dominant_zone for item in items):
        return 0.0
    similarity = _item_similarity_penalty(grid, batch, items)
    return config.same_zone_nearby_boost_weight / max(1.0, similarity)


def _score_batch_insertion(
    grid: GridGraph,
    batch: SmartBatch,
    items: List[BatchItem],
    config: OptimizationConfig,
    picker_speed_mps: float,
) -> tuple[float, List[Coord]]:
    if _would_exceed_zone_limit(batch, items, config):
        return float("inf"), list(batch.route)
    candidate_route, route_increase = _simulate_insertion(grid, batch.route, items)
    route_after = _route_node_distance(grid, candidate_route)
    score = (
        config.dhobr_route_weight * route_increase
        + _zone_mismatch_penalty(batch, items, config)
        + config.dhobr_similarity_weight * _item_similarity_penalty(grid, batch, items)
        + config.dhobr_picker_load_weight * _picker_load_penalty(batch, config)
        + config.dhobr_delay_weight * _delay_penalty(route_after, items, picker_speed_mps)
        - _category_boost(batch, items, config)
        - _same_zone_nearby_boost(grid, batch, items, config)
    )
    return score, candidate_route


def _picker_id_for_new_batch(
    batches: List[SmartBatch],
    config: OptimizationConfig,
    start: Coord,
    first_item: BatchItem,
    grid: GridGraph,
) -> str:
    picker_loads = {
        f"{config.picker_id_prefix}-{idx}": 0
        for idx in range(max(config.employee_count, 1))
    }
    for batch in batches:
        picker_loads[batch.picker_id] = picker_loads.get(batch.picker_id, 0) + batch.current_load
    return min(
        picker_loads,
        key=lambda picker_id: grid.travel_cost(start, first_item.coord) + picker_loads[picker_id],
    )


def _new_batch_score(grid: GridGraph, start: Coord, end: Coord, items: List[BatchItem], config: OptimizationConfig, picker_speed_mps: float) -> tuple[float, List[Coord]]:
    route = [start, end]
    route, route_increase = _simulate_insertion(grid, route, items)
    score = (
        config.dhobr_route_weight * route_increase
        + config.dhobr_delay_weight * _delay_penalty(_route_node_distance(grid, route), items, picker_speed_mps)
        + config.dhobr_new_batch_bias
    )
    return score, route


def _batch_center(targets: List[Coord]) -> tuple[float, float]:
    if not targets:
        return (0.0, 0.0)
    return (
        sum(target[0] for target in targets) / len(targets),
        sum(target[1] for target in targets) / len(targets),
    )


def _spread_penalty(batch_targets: List[Coord], order_targets: List[Coord]) -> float:
    if not batch_targets or not order_targets:
        return 0.0
    center_x, center_y = _batch_center(batch_targets)
    return sum(abs(target[0] - center_x) + abs(target[1] - center_y) for target in order_targets) / len(order_targets)


def _new_aisle_count(batch_targets: List[Coord], order_targets: List[Coord]) -> int:
    existing_aisles = {target[0] for target in batch_targets}
    return len({target[0] for target in order_targets if target[0] not in existing_aisles})


def _seed_score(order: Order, sku_lookup: Dict[str, Coord], grid: GridGraph, depot: Coord) -> tuple[int, float, int]:
    targets = _reachable_batch_targets([order], sku_lookup, grid)
    distances: List[float] = []
    for target in targets:
        try:
            distances.append(grid.travel_cost(depot, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    farthest_distance = max(distances, default=0.0)
    return (order_unit_count(order), farthest_distance, -order.created_at_epoch)


def _batch_feasible(
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
    summary = _summarize_batch(
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
    return not _batch_constraint_violations(summary, config)


def _zone_limit_feasible(batch_orders: List[Order], candidate: Order, zone_lookup: Dict[str, str], config: OptimizationConfig) -> bool:
    zones: Set[str] = set()
    for order in batch_orders:
        zones.update(order_zone_set(order, zone_lookup))
    zones.update(order_zone_set(candidate, zone_lookup))
    return len(zones) <= config.max_zones_per_batch


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
    batch_targets = _reachable_batch_targets(batch_orders, sku_lookup, grid)
    order_targets = _reachable_batch_targets([candidate], sku_lookup, grid)
    new_aisles = _new_aisle_count(batch_targets, order_targets)
    spread_penalty = _spread_penalty(batch_targets, order_targets)
    batch_categories = set().union(*[_order_category_set(order, sku_category_lookup) for order in batch_orders])
    candidate_categories = _order_category_set(candidate, sku_category_lookup)
    category_similarity = _jaccard_similarity(batch_categories, candidate_categories)
    zone_lookup = sku_zone_lookup or sku_category_lookup
    batch_zones = [zone for order in batch_orders for zone in order_zone_set(order, zone_lookup)]
    dominant_zone = Counter(batch_zones).most_common(1)[0][0] if batch_zones else None
    candidate_zones = order_zone_set(candidate, zone_lookup)
    zone_penalty = 0.0
    if dominant_zone is not None and candidate_zones != {dominant_zone}:
        zone_penalty = config.zone_mismatch_penalty_weight * len(candidate_zones - {dominant_zone})
    same_zone_nearby_boost = (
        config.same_zone_nearby_boost_weight / max(1.0, spread_penalty)
        if dominant_zone is not None and candidate_zones == {dominant_zone}
        else 0.0
    )
    return (
        route_delta
        + zone_penalty
        + config.advanced_aisle_penalty_weight * new_aisles
        + config.advanced_spread_penalty_weight * spread_penalty
        - config.advanced_category_boost_weight * category_similarity
        - same_zone_nearby_boost
    )


def _dominant_zone(order: Order, sku_zone_lookup: Dict[str, str]) -> str | None:
    zones = [sku_zone_lookup.get(item.sku, 'unknown') for item in order.items]
    if not zones:
        return None
    return Counter(zones).most_common(1)[0][0]


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
    if not orders:
        return BatchAssignment(batches=[], labels={})

    zone_lookup = sku_zone_lookup or sku_category_lookup
    unassigned = list(orders)
    batches: List[List[Order]] = []
    notes: List[str] = ["advanced_batching=zone_distance_hybrid route_estimator=nearest_neighbor"]

    # Preprocess orders: determine dominant zone and centroid
    order_zones = {order.order_id: _dominant_zone(order, zone_lookup) for order in unassigned}
    order_centroids = {order.order_id: order_centroid(order, sku_lookup) for order in unassigned}

    max_batches = config.employee_count if config.dynamic_batching_enabled else None

    while unassigned:
        if max_batches is not None and len(batches) >= max_batches:
            notes.append(f"batch_count_limit_reached employee_count={config.employee_count} remaining_orders={len(unassigned)}")
            for order in unassigned:
                notes.append(f"unassigned_order {order.order_id}")
            unassigned.clear()
            break

        # Seed selection: order with highest item count or farthest from depot
        seed = max(unassigned, key=lambda order: _seed_score(order, sku_lookup, grid, start))
        unassigned.remove(seed)
        batch = [seed]

        # Greedy expansion
        while unassigned:
            best_order = None
            best_cost = float("inf")

            # Batch properties
            batch_zones = set()
            for order in batch:
                batch_zones.update(order_zone_set(order, zone_lookup))
            batch_centroid = np.mean([order_centroids[o.order_id] for o in batch], axis=0)

            for candidate in unassigned:
                candidate_zones = order_zone_set(candidate, zone_lookup)

                # Hard constraint: max zones per batch
                if len(batch_zones | candidate_zones) > config.max_zones_per_batch:
                    continue

                # Distance cost: increase in route distance
                before_distance = _nearest_neighbor_distance(batch, sku_lookup, grid, start, end)
                after_distance = _nearest_neighbor_distance(batch + [candidate], sku_lookup, grid, start, end)
                distance_cost = after_distance - before_distance

                # Zone penalty: 0 if same zone, high if different
                zone_penalty = 0.0
                if batch_zones and not candidate_zones <= batch_zones:
                    zone_penalty = 1.0  # Will be multiplied by weight

                # Spread penalty: distance from batch centroid
                candidate_centroid = order_centroids[candidate.order_id]
                spread_penalty = np.linalg.norm(batch_centroid - candidate_centroid)

                # Total cost
                total_cost = (
                    config.advanced_route_weight * distance_cost
                    + config.zone_mismatch_penalty_weight * zone_penalty
                    + config.advanced_spread_penalty_weight * spread_penalty
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_order = candidate

            if best_order is None:
                break

            batch.append(best_order)
            unassigned.remove(best_order)

        batches.append(batch)

    # Post-batch optimization: merge batches with same zone
    _merge_singletons(
        batches,
        sku_lookup,
        zone_lookup,
        product_lookup,
        config,
        grid,
        start,
        end,
        picker_speed_mps,
    )
    _merge_same_zone_batches(
        batches,
        sku_lookup,
        zone_lookup,
        product_lookup,
        config,
        grid,
        start,
        end,
        picker_speed_mps,
    )

    # Capacity control and overflow
    standard_limit = min(len(batches), config.employee_count) if config.dynamic_batching_enabled else min(config.batch_count, len(batches))
    batch_names: List[str] = []
    batch_types: List[str] = []
    overflow_idx = 0
    for idx in range(len(batches)):
        if idx < standard_limit:
            batch_names.append(f"{config.batch_id_prefix}-{idx}")
            batch_types.append("standard")
        elif config.allow_overflow_batches:
            batch_names.append(f"{config.overflow_batch_name_prefix}-{overflow_idx}")
            batch_types.append("overflow")
            overflow_idx += 1
        else:
            for order in batches[idx]:
                notes.append(f"unassigned_order {order.order_id}")
            batch_names.append(f"{config.batch_id_prefix}-{idx}")
            batch_types.append("exception")

    # Debug and validation
    for batch_name, batch in zip(batch_names, batches):
        distance = _nearest_neighbor_distance(batch, sku_lookup, grid, start, end)
        total_items = sum(order_unit_count(order) for order in batch)
        order_ids = ",".join(order.order_id for order in batch)
        zones = sorted({zone for order in batch for zone in order_zone_set(order, zone_lookup)})
        zone_counts = Counter(zone for order in batch for zone in order_zone_set(order, zone_lookup))
        dominant_zone = zone_counts.most_common(1)[0][0] if zone_counts else "unassigned"
        notes.append(
            f"Batch ID={batch_name} Total items={total_items} Total distance={distance:.2f} Orders inside=[{order_ids}]"
        )
        notes.append(
            f"Batch ID={batch_name} Zones inside batch={zones} Dominant zone={dominant_zone} Distance={distance:.2f}"
        )

    return _finalize_assignment(
        batches,
        batch_names,
        batch_types,
        notes=notes,
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
            merged_zones = {zone for order in merged for zone in order_zone_set(order, sku_zone_lookup)}
            if len(merged_zones) > config.max_zones_per_batch:
                continue
            if not _batch_feasible(
                merged,
                sku_lookup,
                sku_zone_lookup,
                product_lookup,
                config,
                grid,
                start,
                end,
                picker_speed_mps,
            ):
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
            zones_i = {zone for order in batches[i] for zone in order_zone_set(order, sku_zone_lookup)}
            if len(zones_i) != 1:
                continue
            for j in range(i + 1, len(batches)):
                zones_j = {zone for order in batches[j] for zone in order_zone_set(order, sku_zone_lookup)}
                if zones_i != zones_j:
                    continue
                merged = batches[i] + batches[j]
                if not _batch_feasible(
                    merged,
                    sku_lookup,
                    sku_zone_lookup,
                    product_lookup,
                    config,
                    grid,
                    start,
                    end,
                    picker_speed_mps,
                ):
                    continue
                before = _nearest_neighbor_distance(batches[i], sku_lookup, grid, start, end)
                before += _nearest_neighbor_distance(batches[j], sku_lookup, grid, start, end)
                after = _nearest_neighbor_distance(merged, sku_lookup, grid, start, end)
                if after <= before:
                    batches[i] = merged
                    batches.pop(j)
                    changed = True
                    break


def constrained_k_medoids(
    orders: List[Order],
    sku_lookup: Dict[str, Tuple[int, int]],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> BatchAssignment:
    if not orders:
        return BatchAssignment(batches=[], labels={})
    k = _effective_batch_count(orders, sku_lookup, sku_category_lookup, config)

    features = np.array([order_centroid(o, sku_lookup) for o in orders], dtype=np.float64)
    medoid_indices = _init_medoids(features, k)
    medoids = features[medoid_indices]

    max_iter = 12
    labels = np.zeros(len(orders), dtype=np.int32)
    order_category_sets: List[Set[str]] = [_order_category_set(o, sku_category_lookup) for o in orders]
    overflow_orders: List[Order] = []
    exception_order_ids: List[str] = []
    notes: List[str] = []

    for _ in range(max_iter):
        batch_weights = [0.0 for _ in range(k)]
        batch_sizes = [0 for _ in range(k)]
        batch_category_sets = [set() for _ in range(k)]
        new_labels = np.full(len(orders), -1, dtype=np.int32)
        overflow_orders = []
        exception_order_ids = []

        order_indices = sorted(range(len(orders)), key=lambda i: orders[i].due_time_minutes)
        for i in order_indices:
            order = orders[i]
            weight = order_weight(order)
            costs = []
            for batch_idx in range(k):
                unconstrained_cost = _order_cost(
                    features[i],
                    medoids[batch_idx],
                    order.due_time_minutes,
                    weight,
                    order_category_sets[i],
                    batch_category_sets[batch_idx],
                    config,
                )
                feasible = (
                    batch_sizes[batch_idx] < config.max_batch_size
                    and batch_weights[batch_idx] + weight <= config.max_batch_weight
                )
                costs.append((unconstrained_cost if feasible else float("inf"), batch_idx))

            costs.sort(key=lambda item: item[0])
            if costs and costs[0][0] < float("inf"):
                chosen_batch = costs[0][1]
                new_labels[i] = chosen_batch
                batch_sizes[chosen_batch] += 1
                batch_weights[chosen_batch] += weight
                batch_category_sets[chosen_batch].update(order_category_sets[i])
                continue

            if config.allow_overflow_batches and weight <= config.max_batch_weight:
                overflow_orders.append(order)
            else:
                exception_order_ids.append(order.order_id)

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
            medoid_local_idx = int(np.argmin(pairwise))
            medoids[batch_idx] = member_features[medoid_local_idx]

    standard_batches: List[List[Order]] = [[] for _ in range(k)]
    for idx, order in enumerate(orders):
        batch_idx = int(labels[idx])
        if batch_idx >= 0:
            standard_batches[batch_idx].append(order)

    overflow_batches = [[order] for order in overflow_orders]
    if overflow_batches:
        notes.append("legacy k-medoids placed infeasible leftovers into overflow singleton batches")
    if exception_order_ids:
        notes.append("legacy k-medoids could not assign some orders without violating hard constraints")

    batch_names = [f"{config.batch_id_prefix}-{idx}" for idx in range(len(standard_batches))] + [
        f"{config.overflow_batch_name_prefix}-{idx}" for idx in range(len(overflow_batches))
    ]
    batch_types = ["standard" for _ in standard_batches] + ["overflow" for _ in overflow_batches]
    return _finalize_assignment(
        standard_batches + overflow_batches,
        batch_names,
        batch_types,
        notes=notes,
        exception_order_ids=exception_order_ids,
    )


def greedy_capacity_batching(
    orders: List[Order],
    sku_lookup: Dict[str, Tuple[int, int]],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
) -> BatchAssignment:
    if not orders:
        return BatchAssignment(batches=[], labels={})

    sorted_orders = sorted(orders, key=lambda order: order.due_time_minutes)
    batches: List[List[Order]] = []
    batch_weights: List[float] = []
    max_batches = _effective_batch_count(orders, sku_lookup, sku_category_lookup, config)
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
            group_centroid = np.mean([order_centroid(group_order, sku_lookup) for group_order in group], axis=0)
            cost = float(np.linalg.norm(order_cent - group_centroid))
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        if best_idx == -1 and len(batches) < max_batches:
            batches.append([order])
            batch_weights.append(weight)
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

    batch_names = [f"{config.batch_id_prefix}-{idx}" for idx in range(len(batches))] + [
        f"{config.overflow_batch_name_prefix}-{idx}" for idx in range(len(overflow_batches))
    ]
    batch_types = ["standard" for _ in batches] + ["overflow" for _ in overflow_batches]
    return _finalize_assignment(
        batches + overflow_batches,
        batch_names,
        batch_types,
        notes=notes,
        exception_order_ids=exception_order_ids,
    )
