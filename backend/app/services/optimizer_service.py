from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Literal, Tuple

import networkx as nx

from app.utils.metrics import OptimizationMetricsTracker

from app.ml.travel_time_model import TravelTimeEstimator
from app.optimizer.batching import (
    constrained_k_medoids,
    global_zone_task_batching,
    greedy_capacity_batching,
    insertion_cost_batching,
    seed_distance_batching,
)
from app.optimizer.feature_engineering import (
    build_product_lookup,
    build_sku_category_lookup,
    build_sku_pick_node_lookup,
    build_sku_lookup,
    build_sku_zone_lookup,
    order_pick_nodes,
)
from app.optimizer.graph_model import build_grid_graph
from app.optimizer.reoptimizer import low_disruption_reoptimize
from app.optimizer.routing import (
    expand_route_to_steps,
    route_distance,
    solve_route_nearest_neighbor,
)
from app.schemas import (
    BatchPlan,
    OptimizationRequest,
    OptimizationResponse,
    Order,
    OrderItem,
    RouteStep,
)


OptimizationStrategy = Literal["full", "greedy_nn", "spatial_ortools", "global_zone_task"]
Coord = Tuple[int, int]
logger = logging.getLogger(__name__)


@dataclass
class OptimizationDiagnostics:
    dropped_pick_nodes: int
    total_pick_nodes: int
    late_order_proxy: int
    strategy: str


def _batch_targets(batch_orders, sku_lookup: Dict[str, Coord]) -> List[Coord]:
    nodes: List[Coord] = []
    for order in batch_orders:
        nodes.extend(order_pick_nodes(order, sku_lookup))
    return list(dict.fromkeys(nodes))


def _split_order_into_pick_node_fragments(order, sku_lookup: Dict[str, Coord]) -> List[Order]:
    fragments: Dict[tuple, List[OrderItem]] = {}
    for item in order.items:
        coord = sku_lookup.get(item.sku)
        key = (coord if coord is not None else (-1, -1), item.sku)
        fragments.setdefault(key, []).append(item)

    return [
        Order(
            order_id=order.order_id,
            items=items,
            due_time_minutes=order.due_time_minutes,
            weight_score=order.weight_score,
            created_at_epoch=order.created_at_epoch,
            priority=order.priority,
            latest_pick_start_minutes=order.latest_pick_start_minutes,
            temperature_sensitive=order.temperature_sensitive,
        )
        for items in fragments.values()
    ]


def _expand_orders_for_batching(
    orders: List[Order], 
    sku_lookup: Dict[str, Coord], 
    sku_category_lookup: Dict[str, str],
    allow_order_splitting: bool,
    strict_category_grouping: bool,
) -> List[Order]:
    expanded: List[Order] = []
    for order in orders:
        current_fragments = [order]
        
        if strict_category_grouping:
            cat_fragments = []
            for o in current_fragments:
                cat_dict = {}
                for item in o.items:
                    cat = sku_category_lookup.get(item.sku, "unknown")
                    cat_dict.setdefault(cat, []).append(item)
                for items in cat_dict.values():
                    cat_fragments.append(o.model_copy(update={"items": items}))
            current_fragments = cat_fragments
            
        if allow_order_splitting:
            node_fragments = []
            for o in current_fragments:
                node_dict = {}
                for item in o.items:
                    coord = sku_lookup.get(item.sku)
                    key = (coord if coord is not None else (-1, -1), item.sku)
                    node_dict.setdefault(key, []).append(item)
                for items in node_dict.values():
                    node_fragments.append(o.model_copy(update={"items": items}))
            current_fragments = node_fragments
            
        expanded.extend(current_fragments)
    return expanded


def _reachable_targets(
    grid,
    targets: List[Coord],
    notes: List[str],
    context: str,
) -> Tuple[List[Coord], int]:
    reachable: List[Coord] = []
    dropped = 0
    for target in targets:
        if target not in grid.graph:
            dropped += 1
            continue
        if nx.has_path(grid.graph, grid.entry, target) and nx.has_path(grid.graph, target, grid.exit):
            reachable.append(target)
        else:
            dropped += 1
    if dropped > 0:
        notes.append(f"{context}: dropped {dropped} unreachable pick nodes.")
    return reachable, dropped


def _route_congestion(
    request: OptimizationRequest,
    batch_orders,
    route_nodes: List[Coord],
    sku_zone_lookup: Dict[str, str],
) -> float | None:
    telemetry = request.telemetry
    if telemetry is None:
        return None

    samples: List[float] = []
    if telemetry.global_congestion is not None:
        samples.append(float(telemetry.global_congestion))

    cell_congestion = {
        (reading.cell.x, reading.cell.y): float(reading.level)
        for reading in telemetry.cell_congestion
    }
    samples.extend(cell_congestion[node] for node in route_nodes if node in cell_congestion)

    batch_zones = {
        sku_zone_lookup[item.sku]
        for order in batch_orders
        for item in order.items
        if item.sku in sku_zone_lookup
    }
    samples.extend(
        float(telemetry.zone_congestion[zone])
        for zone in batch_zones
        if zone in telemetry.zone_congestion
    )
    if not samples:
        return None
    return sum(samples) / len(samples)


def _naive_distance(
    request: OptimizationRequest,
    grid,
    sku_lookup: Dict[str, Coord],
    notes: List[str],
) -> Tuple[float, int, int]:
    total = 0.0
    dropped_total = 0
    target_total = 0
    for order in request.orders:
        raw_targets = order_pick_nodes(order, sku_lookup)
        targets, dropped = _reachable_targets(grid, raw_targets, notes, f"naive order {order.order_id}")
        route = solve_route_nearest_neighbor(grid, grid.entry, grid.exit, targets)
        total += route_distance(grid, route)
        dropped_total += dropped
        target_total += len(raw_targets)
    return total, dropped_total, target_total


def _should_use_warehouse_heuristic(request: OptimizationRequest, strategy: OptimizationStrategy, target_count: int) -> bool:
    if strategy == "greedy_nn" or target_count <= 1:
        return False
    if request.config.use_ortools:
        return False
    return bool(
        request.layout.one_way_edges
        or request.layout.turn_penalty > 0
        or request.layout.path_cells
        or request.layout.shelf_cells
    )


def _select_assignment(
    strategy: OptimizationStrategy,
    request: OptimizationRequest,
    sku_lookup: Dict[str, Coord],
    sku_category_lookup: Dict[str, str],
    sku_zone_lookup: Dict[str, str],
    product_lookup,
    grid,
):
    orders_for_batching = _expand_orders_for_batching(
        request.orders, 
        sku_lookup, 
        sku_category_lookup,
        request.config.allow_order_splitting,
        getattr(request.config, "strict_category_grouping", False),
    )

    if strategy == "greedy_nn":
        return greedy_capacity_batching(orders_for_batching, sku_lookup, sku_category_lookup, request.config)
    if strategy == "spatial_ortools":
        spatial_cfg = request.config.model_copy(
            update={"beta_due_time": 0.0, "gamma_weight": 0.0, "delta_similarity": 0.0}
        )
        return constrained_k_medoids(orders_for_batching, sku_lookup, sku_category_lookup, spatial_cfg)
    if strategy == "global_zone_task":
        return global_zone_task_batching(
            orders_for_batching,
            sku_lookup,
            sku_category_lookup,
            request.config,
            sku_zone_lookup=sku_zone_lookup,
            product_lookup=product_lookup,
            grid=grid,
            start=grid.entry,
            end=grid.exit,
            picker_speed_mps=request.picker_speed_mps,
        )
    if request.config.use_insertion_batching:
        return insertion_cost_batching(
            orders_for_batching,
            sku_lookup,
            sku_category_lookup,
            request.config,
            sku_zone_lookup=sku_zone_lookup,
            product_lookup=product_lookup,
            grid=grid,
            start=grid.entry,
            end=grid.exit,
            picker_speed_mps=request.picker_speed_mps,
        )
    return constrained_k_medoids(orders_for_batching, sku_lookup, sku_category_lookup, request.config)


def _validate_route_nodes(grid, route_nodes: List[Coord], batch_id: str) -> None:
    for node in route_nodes:
        if node not in grid.graph:
            raise ValueError(f"{batch_id}: route contains non-walkable node {node}")
    for idx in range(len(route_nodes) - 1):
        path = grid.shortest_path(route_nodes[idx], route_nodes[idx + 1])
        blocked_crossing = [node for node in path if node in grid.blocked]
        if blocked_crossing:
            raise ValueError(f"{batch_id}: route crosses blocked cells {blocked_crossing}")


def optimize_orders_with_strategy(
    request: OptimizationRequest,
    estimator: TravelTimeEstimator,
    strategy: OptimizationStrategy = "full",
    existing_plans: List[BatchPlan] | None = None,
) -> Tuple[OptimizationResponse, OptimizationDiagnostics]:
    tracker = OptimizationMetricsTracker()
    start = time.perf_counter()
    grid = build_grid_graph(request.layout)
    execution_state = request.reoptimization
    product_lookup = build_product_lookup(request.product_map)
    raw_sku_lookup = build_sku_lookup(request.product_map)
    route_sku_lookup = build_sku_pick_node_lookup(
        request.product_map,
        enable_pick_face=request.config.enable_pick_face_routing,
        walkable_nodes=grid.graph.nodes,
    )
    sku_category_lookup = build_sku_category_lookup(request.product_map)
    sku_zone_lookup = build_sku_zone_lookup(request.product_map)
    notes: List[str] = []

    if strategy == "full":
        orders_for_batching = _expand_orders_for_batching(
            request.orders, 
            route_sku_lookup, 
            sku_category_lookup,
            False, # Disable pick-node splitting for realistic zone-based batching
            getattr(request.config, "strict_category_grouping", False),
        )
        with tracker.phase("batching"):
            assignment = seed_distance_batching(
                orders_for_batching,
                route_sku_lookup,
                sku_category_lookup,
                request.config,
                sku_zone_lookup=sku_zone_lookup,
                product_lookup=product_lookup,
                grid=grid,
                start=grid.entry,
                end=grid.exit,
                picker_speed_mps=request.picker_speed_mps,
            )
    else:
        with tracker.phase("batching"):
            assignment = _select_assignment(
                strategy,
                request,
                route_sku_lookup,
                sku_category_lookup,
                sku_zone_lookup,
                product_lookup,
                grid,
            )
    notes.extend(assignment.notes)
    batch_plans: List[BatchPlan] = []
    dropped_total = 0
    target_total = 0
    overflow_batch_ids: List[str] = []

    for idx, orders in enumerate(assignment.batches):
        batch_id = assignment.batch_names[idx] if idx < len(assignment.batch_names) else f"{request.config.batch_id_prefix}-{idx}"
        batch_type = assignment.batch_types[idx] if idx < len(assignment.batch_types) else "standard"
        raw_targets = _batch_targets(orders, route_sku_lookup)
        targets, dropped = _reachable_targets(grid, raw_targets, notes, batch_id)
        dropped_total += dropped
        target_total += len(raw_targets)
        if not targets and raw_targets:
            notes.append(f"{batch_id}: all pick nodes were unreachable; routing direct entry-to-exit path.")

        route_nodes = solve_route_nearest_neighbor(grid, grid.entry, grid.exit, targets)
        _validate_route_nodes(grid, route_nodes, batch_id)
        distance = route_distance(grid, route_nodes)
        congestion = _route_congestion(request, orders, route_nodes, sku_zone_lookup)
        est_seconds = estimator.predict_seconds_hybrid(
            distance=distance,
            congestion=congestion,
            cart_load=sum(item.qty for order in orders for item in order.items),
            picker_speed=request.picker_speed_mps,
            stop_count=len(targets),
        )
        order_ids = sorted({order.order_id for order in orders})
        route_steps = [
            RouteStep(**step)
            for step in expand_route_to_steps(grid, route_nodes, route_sku_lookup, order_ids)
        ]
        picked_skus = sorted({item.sku for order in orders for item in order.items if item.sku in raw_sku_lookup})
        if batch_type == "overflow":
            overflow_batch_ids.append(batch_id)
        batch_plans.append(
            BatchPlan(
                batch_id=batch_id,
                order_ids=order_ids,
                picked_skus=picked_skus,
                distance=distance,
                estimated_seconds=est_seconds,
                route=route_steps,
            )
        )

    prior_plans = existing_plans
    if prior_plans is None and execution_state and execution_state.existing_batch_plans:
        prior_plans = execution_state.existing_batch_plans

    if request.config.allow_dynamic_reoptimization and prior_plans is not None:
        reopt = low_disruption_reoptimize(
            prior_plans,
            batch_plans,
            request.config,
            batch_statuses=execution_state.batch_statuses if execution_state else None,
            current_picker_positions=(
                {batch_id: (cell.x, cell.y) for batch_id, cell in execution_state.current_picker_positions.items()}
                if execution_state
                else None
            ),
            locked_order_ids=execution_state.locked_order_ids if execution_state else None,
            completed_order_ids=execution_state.completed_order_ids if execution_state else None,
        )
        batch_plans = reopt.plans
        notes.append(f"Dynamic reoptimization instability_cost={reopt.instability_cost:.2f}")

    batched_distance = sum(plan.distance for plan in batch_plans)
    batched_time = sum(plan.estimated_seconds for plan in batch_plans)
    naive_distance, naive_dropped, naive_targets = _naive_distance(request, grid, route_sku_lookup, notes)
    naive_time = naive_distance / max(request.picker_speed_mps, 0.1)
    dropped_pick_rate = (dropped_total / target_total) if target_total > 0 else 0.0
    naive_dropped_rate = (naive_dropped / naive_targets) if naive_targets > 0 else 0.0
    order_to_eta_minutes: Dict[str, float] = {}
    for batch_plan in batch_plans:
        eta_minutes = batch_plan.estimated_seconds / 60.0
        for order_id in batch_plan.order_ids:
            order_to_eta_minutes[order_id] = eta_minutes
    late_order_proxy = sum(
        1
        for order in request.orders
        if order.order_id in assignment.exception_order_ids
        or order_to_eta_minutes.get(order.order_id, float("inf")) > order.due_time_minutes
    )

    runtime_ms = (time.perf_counter() - start) * 1000
    improvement_pct = ((naive_distance - batched_distance) / naive_distance * 100.0) if naive_distance > 0 else 0.0
    tracker.record("improvement_pct", improvement_pct)
    tracker.record("naive_distance", naive_distance)
    tracker.record("batched_distance", batched_distance)
    tracker.record("dropped_pick_rate", dropped_pick_rate)
    tracker.emit({"strategy": strategy, "order_count": len(request.orders), "batch_count": len(batch_plans)})
    prediction_eval = estimator.get_evaluation() or {}
    eval_method = prediction_eval.get("evaluation_method", "not_available")
    eval_r2 = prediction_eval.get("r2", "n/a")
    eval_mae = prediction_eval.get("mae", "n/a")
    eval_rmse = prediction_eval.get("rmse", "n/a")

    if overflow_batch_ids:
        notes.append(f"overflow_batches={len(overflow_batch_ids)}")
    if assignment.exception_order_ids:
        notes.append(f"unassigned_orders={len(assignment.exception_order_ids)}")

    response = OptimizationResponse(
        metrics={
            "naive_distance": naive_distance,
            "batched_distance": batched_distance,
            "improvement_pct": improvement_pct,
            "naive_time_seconds": naive_time,
            "batched_time_seconds": batched_time,
            "runtime_ms": runtime_ms,
            "prediction_eval_method": prediction_eval.get("evaluation_method"),
            "prediction_r2": prediction_eval.get("r2"),
            "prediction_mae": prediction_eval.get("mae"),
            "prediction_rmse": prediction_eval.get("rmse"),
            "prediction_cv_r2_mean": prediction_eval.get("cv_r2_mean"),
            "prediction_cv_mae_mean": prediction_eval.get("cv_mae_mean"),
            "prediction_cv_rmse_mean": prediction_eval.get("cv_rmse_mean"),
        },
        batch_plans=batch_plans,
        cluster_labels=assignment.labels,
        notes=notes
        + [
            f"strategy={strategy}",
            f"late_order_proxy={late_order_proxy}",
            f"dropped_pick_rate={dropped_pick_rate:.4f}",
            f"naive_dropped_pick_rate={naive_dropped_rate:.4f}",
            f"prediction_eval={eval_method} R2={eval_r2} MAE={eval_mae} RMSE={eval_rmse}",
        ],
        overflow_batch_ids=overflow_batch_ids,
        unassigned_order_ids=assignment.exception_order_ids,
    )
    diagnostics = OptimizationDiagnostics(
        dropped_pick_nodes=dropped_total,
        total_pick_nodes=target_total,
        late_order_proxy=late_order_proxy,
        strategy=strategy,
    )
    return response, diagnostics


def optimize_orders(
    request: OptimizationRequest,
    estimator: TravelTimeEstimator,
    existing_plans: List[BatchPlan] | None = None,
) -> OptimizationResponse:
    response, _ = optimize_orders_with_strategy(
        request,
        estimator=estimator,
        strategy="full",
        existing_plans=existing_plans,
    )
    return response
