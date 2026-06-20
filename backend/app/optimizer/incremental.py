from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import random
import time
from typing import Dict, List, Sequence, Tuple

import networkx as nx

from app.optimizer.batching import (
    BatchAssignment,
    _batch_constraint_violations,
    _effective_batch_count,
    _finalize_assignment,
    _summarize_batch,
)
from app.optimizer.feature_engineering import (
    order_fragility_flags,
    order_pick_nodes,
    order_unit_count,
    order_volume,
    order_weight,
    order_zone_set,
)
from app.optimizer.graph_model import GridGraph
from app.optimizer.routing import route_distance
from app.schemas import OptimizationConfig, Order


Coord = Tuple[int, int]
BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="route-ils")
logger = logging.getLogger(__name__)


@dataclass
class ActiveBatch:
    batch_id: str
    orders: List[Order] = field(default_factory=list)
    route: List[Coord] = field(default_factory=list)
    current_load: int = 0
    current_weight: float = 0.0
    current_volume: float = 0.0
    assigned_picker: str | None = None
    batch_type: str = "standard"


@dataclass
class IncrementalBatchingResult:
    assignment: BatchAssignment
    routes: Dict[str, List[Coord]]
    active_batches: List[ActiveBatch]


def _batch_assignment_from_active(
    active_batches: Sequence[ActiveBatch],
    notes: Sequence[str],
    exception_order_ids: Sequence[str],
) -> BatchAssignment:
    return _finalize_assignment(
        [batch.orders for batch in active_batches],
        [batch.batch_id for batch in active_batches],
        [batch.batch_type for batch in active_batches],
        notes=list(notes),
        exception_order_ids=list(exception_order_ids),
    )


def _route_insertion_delta(grid: GridGraph, route: Sequence[Coord], node: Coord, pos: int) -> float:
    prev_node = route[pos - 1]
    next_node = route[pos]
    return grid.travel_cost(prev_node, node) + grid.travel_cost(node, next_node) - grid.travel_cost(prev_node, next_node)


def best_insert_node(grid: GridGraph, route: Sequence[Coord], node: Coord) -> tuple[List[Coord], float]:
    if node in route:
        return list(route), 0.0
    if len(route) < 2:
        return list(route) + [node], 0.0

    best_pos = 1
    best_delta = float("inf")
    for pos in range(1, len(route)):
        delta = _route_insertion_delta(grid, route, node, pos)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    updated = list(route)
    updated.insert(best_pos, node)
    return updated, best_delta


def insert_targets(
    grid: GridGraph,
    route: Sequence[Coord],
    targets: Sequence[Coord],
    *,
    stability_threshold: float = 0.0,
) -> tuple[List[Coord], float]:
    updated = list(route)
    total_delta = 0.0
    for target in dict.fromkeys(targets):
        candidate, delta = best_insert_node(grid, updated, target)
        if target in updated or delta >= stability_threshold:
            updated = candidate
            total_delta += delta
    return updated, total_delta


def two_opt(grid: GridGraph, route: Sequence[Coord], *, max_passes: int = 2, min_gain: float = 0.01) -> List[Coord]:
    best = list(route)
    if len(best) <= 4:
        return best
    try:
        best_distance = route_distance(grid, best)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return best
    for _ in range(max_passes):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                try:
                    candidate_distance = route_distance(grid, candidate)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if best_distance - candidate_distance > min_gain:
                    best = candidate
                    best_distance = candidate_distance
                    improved = True
        if not improved:
            break
    return best


def iterated_local_search(
    grid: GridGraph,
    route: Sequence[Coord],
    *,
    iterations: int,
    stability_threshold: float,
    random_seed: int,
    two_opt_passes: int,
    min_gain: float,
) -> List[Coord]:
    if len(route) <= 4:
        return list(route)
    rng = random.Random(random_seed)
    best = two_opt(grid, route, max_passes=two_opt_passes, min_gain=min_gain)
    best_distance = route_distance(grid, best)
    for _ in range(iterations):
        candidate = list(best)
        if len(candidate) > 5:
            i, j = sorted(rng.sample(range(1, len(candidate) - 1), 2))
            candidate[i], candidate[j] = candidate[j], candidate[i]
        if len(candidate) > 4:
            src = rng.randrange(1, len(candidate) - 1)
            node = candidate.pop(src)
            dst = rng.randrange(1, len(candidate))
            candidate.insert(dst, node)
        candidate = two_opt(grid, candidate, max_passes=two_opt_passes, min_gain=min_gain)
        try:
            candidate_distance = route_distance(grid, candidate)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if best_distance - candidate_distance >= stability_threshold:
            best = candidate
            best_distance = candidate_distance
    return best


def improve_routes_background(
    grid: GridGraph,
    routes: Dict[str, List[Coord]],
    *,
    config: OptimizationConfig,
) -> None:
    def _work() -> None:
        for batch_id, route in routes.items():
            decision_start_distance = route_distance(grid, route)
            improved = iterated_local_search(
                grid,
                route,
                iterations=config.ils_iterations,
                stability_threshold=config.route_improvement_threshold,
                random_seed=config.ils_random_seed,
                two_opt_passes=config.ils_two_opt_passes,
                min_gain=config.route_local_search_min_gain,
            )
            try:
                improvement = decision_start_distance - route_distance(grid, improved)
                if improvement >= config.route_improvement_threshold:
                    routes[batch_id] = improved
                    logger.info(
                        "background_route_improved batch_id=%s improvement=%.3f route_nodes=%s",
                        batch_id,
                        improvement,
                        len(improved),
                    )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    BACKGROUND_EXECUTOR.submit(_work)


class IncrementalBatcher:
    def __init__(
        self,
        *,
        grid: GridGraph,
        sku_lookup: Dict[str, Coord],
        sku_category_lookup: Dict[str, str],
        sku_zone_lookup: Dict[str, str],
        product_lookup,
        config: OptimizationConfig,
        picker_speed_mps: float,
    ) -> None:
        self.grid = grid
        self.sku_lookup = sku_lookup
        self.sku_category_lookup = sku_category_lookup
        self.sku_zone_lookup = sku_zone_lookup or sku_category_lookup
        self.product_lookup = product_lookup
        self.config = config
        self.picker_speed_mps = picker_speed_mps
        self.target_standard_batches = 0
        self.hard_standard_batch_cap = 0
        self.overflow_counter = 0
        self.notes: List[str] = []
        self.exception_order_ids: List[str] = []
        self.active_batches: List[ActiveBatch] = []
        self.total_decisions = 0
        self.total_route_changes = 0

    def set_order_horizon(self, orders: Sequence[Order]) -> None:
        self.target_standard_batches = _effective_batch_count(
            list(orders),
            self.sku_lookup,
            self.sku_category_lookup,
            self.config,
        )
        self.hard_standard_batch_cap = self.target_standard_batches if not self.config.dynamic_batching_enabled else min(self.config.employee_count, len(orders))

    def process(self, orders: List[Order]) -> IncrementalBatchingResult:
        self.set_order_horizon(orders)
        for order in orders:
            self.on_new_order(order)
        return self.result()

    def result(self) -> IncrementalBatchingResult:
        routes = {batch.batch_id: list(batch.route) for batch in self.active_batches}
        assignment = _batch_assignment_from_active(self.active_batches, self.notes, self.exception_order_ids)
        assignment.notes.append("incremental_batching=active_batches route_update=insertion")
        return IncrementalBatchingResult(assignment=assignment, routes=routes, active_batches=list(self.active_batches))

    def on_new_order(self, order: Order) -> ActiveBatch | None:
        decision_start = time.perf_counter()
        self.total_decisions += 1
        single_summary = _summarize_batch(
            [order],
            self.sku_lookup,
            self.sku_zone_lookup,
            self.product_lookup,
            self.config,
            grid=self.grid,
            start=self.grid.entry,
            end=self.grid.exit,
            picker_speed_mps=self.picker_speed_mps,
        )
        standalone_violations = _batch_constraint_violations(single_summary, self.config)
        if standalone_violations:
            self.exception_order_ids.append(order.order_id)
            self.notes.append(f"order {order.order_id} infeasible as standalone batch: {', '.join(standalone_violations)}")
            return None

        targets = self._reachable_order_targets(order)
        best_batch: ActiveBatch | None = None
        best_route: List[Coord] = []
        best_cost = float("inf")
        best_delta = 0.0

        for batch in self.active_batches:
            candidate_orders = batch.orders + [order]
            candidate_summary = _summarize_batch(
                candidate_orders,
                self.sku_lookup,
                self.sku_zone_lookup,
                self.product_lookup,
                self.config,
                grid=self.grid,
                start=self.grid.entry,
                end=self.grid.exit,
                picker_speed_mps=self.picker_speed_mps,
            )
            violations = _batch_constraint_violations(candidate_summary, self.config)
            if violations:
                continue
            candidate_route, route_delta = insert_targets(self.grid, batch.route, targets)
            lateness_penalty = max(0.0, (candidate_summary.duration_seconds / 60.0) - order.due_time_minutes)
            capacity_penalty = self._capacity_pressure(candidate_summary)
            zone_penalty = 1.0
            order_zones = order_zone_set(order, self.sku_zone_lookup)
            if not order_zones or order_zones & candidate_summary.zones:
                zone_penalty = 0.0
            fragility_penalty = self._fragility_penalty(order, batch.orders)
            cost = (
                self.config.alpha_distance * route_delta
                + self.config.beta_due_time * lateness_penalty
                + self.config.gamma_weight * capacity_penalty
                + self.config.delta_similarity * zone_penalty
                + fragility_penalty
            )
            if cost < best_cost:
                best_batch = batch
                best_route = candidate_route
                best_cost = cost
                best_delta = route_delta

        if best_batch is not None and not self._should_create_stable_new_batch(order, best_batch, best_delta):
            self._append_order(best_batch, order, best_route)
            logger.info(
                "order_assigned_incrementally order_id=%s batch_id=%s cost=%.3f route_delta=%.3f decision_ms=%.3f batches=%s",
                order.order_id,
                best_batch.batch_id,
                best_cost,
                best_delta,
                (time.perf_counter() - decision_start) * 1000,
                len(self.active_batches),
            )
            return best_batch

        return self._create_batch_for(order, targets)

    def _append_order(self, batch: ActiveBatch, order: Order, route: List[Coord]) -> None:
        batch.orders.append(order)
        previous_route = list(batch.route)
        batch.route = route if route_distance(self.grid, route) >= 0 else batch.route
        if batch.route != previous_route:
            self.total_route_changes += 1
        batch.current_load += order_unit_count(order)
        batch.current_weight += order_weight(order, self.product_lookup)
        batch.current_volume += order_volume(order, self.product_lookup)

    def _create_batch_for(self, order: Order, targets: Sequence[Coord]) -> ActiveBatch | None:
        standard_count = sum(1 for batch in self.active_batches if batch.batch_type == "standard")
        if standard_count < self.target_standard_batches or standard_count < self.hard_standard_batch_cap:
            batch_id = f"{self.config.batch_id_prefix}-{standard_count}"
            batch_type = "standard"
        elif self.config.allow_overflow_batches:
            batch_id = f"{self.config.overflow_batch_name_prefix}-{self.overflow_counter}"
            batch_type = "overflow"
            self.overflow_counter += 1
            self.notes.append(f"order {order.order_id} assigned to overflow batch {batch_id}")
        else:
            self.exception_order_ids.append(order.order_id)
            self.notes.append(f"order {order.order_id} could not be assigned without violating hard constraints")
            return None

        route, _ = insert_targets(self.grid, [self.grid.entry, self.grid.exit], targets)
        batch = ActiveBatch(
            batch_id=batch_id,
            orders=[order],
            route=route,
            current_load=order_unit_count(order),
            current_weight=order_weight(order, self.product_lookup),
            current_volume=order_volume(order, self.product_lookup),
            assigned_picker=f"{self.config.picker_id_prefix}-{len(self.active_batches) % max(self.config.employee_count, 1)}",
            batch_type=batch_type,
        )
        self.active_batches.append(batch)
        self.total_route_changes += 1
        logger.info(
            "batch_created_incrementally order_id=%s batch_id=%s batch_type=%s distance=%.3f batches=%s",
            order.order_id,
            batch.batch_id,
            batch.batch_type,
            route_distance(self.grid, batch.route),
            len(self.active_batches),
        )
        return batch

    def _should_create_stable_new_batch(self, order: Order, batch: ActiveBatch, route_delta: float) -> bool:
        standard_count = sum(1 for active in self.active_batches if active.batch_type == "standard")
        if standard_count >= self.target_standard_batches or batch.batch_type != "standard":
            return False
        order_targets = self._reachable_order_targets(order)
        if not order_targets:
            return False
        existing_targets = [node for grouped in batch.orders for node in order_pick_nodes(grouped, self.sku_lookup)]
        if not existing_targets:
            return False
        reachable_existing = [node for node in existing_targets if node in self.grid.graph]
        if not reachable_existing or order_targets[0] not in self.grid.graph:
            return False
        try:
            nearest_existing = min(self.grid.travel_cost(order_targets[0], node) for node in reachable_existing)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
        dynamic_threshold = max(
            self.config.stable_new_batch_min_distance,
            nearest_existing * self.config.stable_new_batch_distance_ratio,
        )
        return route_delta > dynamic_threshold

    def _reachable_order_targets(self, order: Order) -> List[Coord]:
        targets: List[Coord] = []
        for target in order_pick_nodes(order, self.sku_lookup):
            if target not in self.grid.graph:
                continue
            if nx.has_path(self.grid.graph, self.grid.entry, target) and nx.has_path(self.grid.graph, target, self.grid.exit):
                targets.append(target)
        return list(dict.fromkeys(targets))

    def _capacity_pressure(self, summary) -> float:
        pressure = summary.order_count / max(self.config.max_batch_size, 1)
        pressure += summary.total_weight / max(self.config.max_batch_weight, self.config.min_capacity_denominator)
        if self.config.max_batch_volume:
            pressure += summary.total_volume / self.config.max_batch_volume
        pressure += summary.target_count / max(self.config.max_shelf_visits_per_picker, 1)
        return pressure

    def _fragility_penalty(self, order: Order, batch_orders: List[Order]) -> float:
        order_flags = order_fragility_flags(order, self.product_lookup)
        batch_flags = [order_fragility_flags(batch_order, self.product_lookup) for batch_order in batch_orders]
        if order_flags["fragile"] and any(flags["bulky"] for flags in batch_flags):
            return self.config.fragile_bulky_penalty
        if order_flags["bulky"] and any(flags["fragile"] for flags in batch_flags):
            return self.config.fragile_bulky_penalty
        return 0.0


def incremental_batching(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    config: OptimizationConfig,
    *,
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    grid: GridGraph,
    picker_speed_mps: float,
) -> IncrementalBatchingResult:
    batcher = IncrementalBatcher(
        grid=grid,
        sku_lookup=sku_lookup,
        sku_category_lookup=sku_category_lookup,
        sku_zone_lookup=sku_zone_lookup,
        product_lookup=product_lookup,
        config=config,
        picker_speed_mps=picker_speed_mps,
    )
    return batcher.process(orders)
