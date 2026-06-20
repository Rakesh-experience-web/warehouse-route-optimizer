"""global_zone_task.py — Territory-aware zone-task batching strategy.

This module implements a fundamentally different warehouse batching paradigm:
instead of assigning whole orders to pickers, it decomposes every order into
per-SKU **PickTask** objects, assigns tasks to pickers based on zone/shelf
ownership, and reconstructs partial orders so that the downstream system
(summaries, metrics, API responses) continues to work unchanged.

Key innovations over order-centric batching:
  - Pickers specialise in warehouse zones (territory ownership).
  - Duplicate shelf visits across pickers are minimised.
  - Global residual capacity is exploited before opening new pickers.
  - Strict 25-item picker capacity is enforced at all times.
  - Route-aware scoring penalises cross-zone detours.

Public function:
  - global_zone_task_batching(...)  — drop-in for the strategy dispatcher.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx

from app.optimizer.batching._types import BatchAssignment
from app.optimizer.batching.assignment import finalize_assignment
from app.optimizer.batching._summary import reachable_batch_targets
from app.optimizer.feature_engineering import order_unit_count
from app.optimizer.graph_model import GridGraph
from app.optimizer.routing import route_distance, solve_route_nearest_neighbor
from app.schemas import OptimizationConfig, Order, OrderItem

Coord = Tuple[int, int]
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STEP 2 — PickTask data model
# ---------------------------------------------------------------------------

@dataclass
class PickTask:
    """A single SKU-level pick task decomposed from a customer order."""

    order_id: str
    sku: str
    qty: int
    zone: str
    coord: Coord


# ---------------------------------------------------------------------------
# STEP 3 — Task route estimation
# ---------------------------------------------------------------------------

def _task_route_distance(
    tasks: List[PickTask],
    grid: GridGraph,
    start: Coord,
    end: Coord,
) -> float:
    """Estimate the pick-route distance for a set of tasks.

    Re-uses the existing nearest-neighbour solver and route distance
    calculator so that all strategies share the same routing semantics.
    """
    if not tasks:
        try:
            return grid.travel_cost(start, end)
        except Exception:
            return 0.0

    # Deduplicate coordinates and keep only reachable ones.
    raw_targets = list(dict.fromkeys(t.coord for t in tasks))
    targets: List[Coord] = []
    for target in raw_targets:
        if target not in grid.graph:
            continue
        try:
            if (
                nx.has_path(grid.graph, start, target)
                and nx.has_path(grid.graph, target, end)
            ):
                targets.append(target)
        except nx.NodeNotFound:
            continue

    if not targets:
        try:
            return grid.travel_cost(start, end)
        except Exception:
            return 0.0

    route = solve_route_nearest_neighbor(grid, start, end, targets)
    return route_distance(grid, route)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _Picker:
    """Mutable live state for a picker during task assignment."""

    picker_id: str
    tasks: List[PickTask] = field(default_factory=list)
    zones: Set[str] = field(default_factory=set)
    capacity_used: int = 0
    primary_zone: str | None = None

    def can_accept(self, task: PickTask, max_units: int) -> bool:
        """Return True when adding *task* would not violate capacity."""
        return self.capacity_used + task.qty <= max_units


def _assignment_score(
    picker: _Picker,
    task: PickTask,
    route_delta: float,
    *,
    territory_bonus: float = 5.0,
    utilization_bonus_weight: float = 2.0,
    zone_mismatch_penalty: float = 8.0,
    max_units: int = 25,
) -> float:
    """Score a candidate picker for a given task (lower is better).

    Components:
      + territory bonus     — reward when task zone matches primary zone
      + utilization bonus   — prefer pickers already partially loaded
      - route delta         — penalise route increase
      - zone mismatch       — penalise cross-zone expansion
    """
    score = route_delta

    # Territory bonus (negative because lower score = better).
    if picker.primary_zone is not None and picker.primary_zone == task.zone:
        score -= territory_bonus

    # Utilization bonus: prefer filling partially-loaded pickers.
    utilisation_pct = picker.capacity_used / max(max_units, 1)
    score -= utilization_bonus_weight * utilisation_pct

    # Zone mismatch penalty.
    if task.zone not in picker.zones and picker.zones:
        score += zone_mismatch_penalty

    return score


# ---------------------------------------------------------------------------
# STEP 5 — Order → Task conversion
# ---------------------------------------------------------------------------

def _decompose_orders_to_tasks(
    orders: List[Order],
    sku_lookup: Dict[str, Coord],
    sku_zone_lookup: Dict[str, str],
) -> List[PickTask]:
    """Convert every order-item into a PickTask."""
    tasks: List[PickTask] = []
    for order in orders:
        for item in order.items:
            coord = sku_lookup.get(item.sku)
            if coord is None:
                continue
            zone = sku_zone_lookup.get(item.sku, "unknown")
            tasks.append(
                PickTask(
                    order_id=order.order_id,
                    sku=item.sku,
                    qty=item.qty,
                    zone=zone,
                    coord=coord,
                )
            )
    return tasks


# ---------------------------------------------------------------------------
# STEP 12 — Order reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_orders(
    picker: _Picker,
    original_orders: Dict[str, Order],
) -> List[Order]:
    """Rebuild partial Order objects from the tasks assigned to a picker.

    Only the SKUs actually assigned to this picker are kept; all other
    order metadata is cloned from the original order so that downstream
    systems (summaries, batch plans, API) remain compatible.
    """
    # Group tasks by order_id.
    sku_map: Dict[str, List[PickTask]] = defaultdict(list)
    for task in picker.tasks:
        sku_map[task.order_id].append(task)

    reconstructed: List[Order] = []
    for order_id, tasks in sku_map.items():
        original = original_orders.get(order_id)
        if original is None:
            continue

        # Build new item list with only the assigned SKUs.
        assigned_skus = {t.sku: t.qty for t in tasks}
        new_items: List[OrderItem] = []
        for item in original.items:
            if item.sku in assigned_skus:
                new_items.append(
                    OrderItem(sku=item.sku, qty=assigned_skus[item.sku])
                )

        if not new_items:
            continue

        cloned = original.model_copy(update={"items": new_items})
        reconstructed.append(cloned)

    return reconstructed


# ---------------------------------------------------------------------------
# STEP 4 / Main algorithm — global_zone_task_batching
# ---------------------------------------------------------------------------

def global_zone_task_batching(
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
    """Territory-aware zone-task batching strategy.

    Signature intentionally matches the existing batching functions so that
    it can be dispatched from ``batch_builder.py`` without adapter code.
    """
    if not orders:
        return BatchAssignment(batches=[], labels={})

    zone_lookup = sku_zone_lookup or sku_category_lookup
    max_units: int = getattr(config, "max_batch_units", 25)
    notes: List[str] = [
        "strategy=global_zone_task_batching",
        "paradigm=task_decomposition_territory_ownership",
    ]

    # ---- STEP 5: Decompose orders into PickTasks ----
    all_tasks = _decompose_orders_to_tasks(orders, sku_lookup, zone_lookup)
    if not all_tasks:
        notes.append("no_valid_tasks_after_decomposition")
        return BatchAssignment(batches=[], labels={}, notes=notes)

    original_orders: Dict[str, Order] = {o.order_id: o for o in orders}
    total_task_qty = sum(t.qty for t in all_tasks)
    notes.append(f"decomposed_tasks={len(all_tasks)} total_qty={total_task_qty}")

    # ---- STEP 6: Group tasks by zone ----
    zone_tasks: Dict[str, List[PickTask]] = defaultdict(list)
    for task in all_tasks:
        zone_tasks[task.zone].append(task)

    zone_summary = {z: sum(t.qty for t in ts) for z, ts in zone_tasks.items()}
    notes.append(f"zones={dict(zone_summary)}")

    # ---- STEP 7: Picker initialisation ----
    num_pickers = config.employee_count
    pickers: List[_Picker] = [
        _Picker(picker_id=f"{config.picker_id_prefix}-{i}")
        for i in range(num_pickers)
    ]

    # ---- STEP 8: Territory ownership ----
    # Assign primary zones to pickers round-robin by descending zone volume.
    sorted_zones = sorted(zone_tasks.keys(), key=lambda z: zone_summary[z], reverse=True)
    for idx, zone in enumerate(sorted_zones):
        picker_idx = idx % num_pickers
        pickers[picker_idx].primary_zone = pickers[picker_idx].primary_zone or zone
        pickers[picker_idx].zones.add(zone)
    notes.append(
        "territory_assignment="
        + " ".join(f"{p.picker_id}:{p.primary_zone}" for p in pickers if p.primary_zone)
    )

    # ---- STEP 10: Territory-first task assignment ----
    # Process zones in descending volume order so that high-volume zones
    # fill their owning pickers first.
    unassigned_tasks: List[PickTask] = []

    for zone in sorted_zones:
        tasks = zone_tasks[zone]
        # Sort tasks within the zone: group by shelf coord to minimise
        # duplicate shelf visits, then by order_id for determinism.
        tasks.sort(key=lambda t: (t.coord, t.order_id))

        for task in tasks:
            best_picker: _Picker | None = None
            best_score = float("inf")

            for picker in pickers:
                # STEP 9: Strict capacity enforcement — never exceed max_units.
                if not picker.can_accept(task, max_units):
                    continue

                # Calculate route delta (approximate: avoid expensive full
                # recalculation on every candidate).
                if picker.tasks:
                    current_dist = _task_route_distance(picker.tasks, grid, start, end)
                    candidate_dist = _task_route_distance(
                        picker.tasks + [task], grid, start, end
                    )
                    route_delta = candidate_dist - current_dist
                else:
                    route_delta = 0.0

                score = _assignment_score(
                    picker, task, route_delta,
                    max_units=max_units,
                )
                if score < best_score:
                    best_score = score
                    best_picker = picker

            if best_picker is not None:
                best_picker.tasks.append(task)
                best_picker.capacity_used += task.qty
                best_picker.zones.add(task.zone)
            else:
                unassigned_tasks.append(task)

    # ---- STEP 11: Global residual capacity optimisation ----
    # Attempt to place overflow tasks into existing pickers with residual
    # capacity before spawning new pickers.
    still_unassigned: List[PickTask] = []
    max_acceptable_detour = 20.0  # warehouse-grid units

    for task in unassigned_tasks:
        placed = False
        # Sort pickers by remaining capacity (most room first) to
        # distribute evenly.
        candidates = sorted(pickers, key=lambda p: (max_units - p.capacity_used), reverse=True)
        for picker in candidates:
            if not picker.can_accept(task, max_units):
                continue
            if picker.tasks:
                current_dist = _task_route_distance(picker.tasks, grid, start, end)
                candidate_dist = _task_route_distance(
                    picker.tasks + [task], grid, start, end
                )
                detour = candidate_dist - current_dist
                if detour > max_acceptable_detour:
                    continue
            picker.tasks.append(task)
            picker.capacity_used += task.qty
            picker.zones.add(task.zone)
            placed = True
            break

        if not placed:
            still_unassigned.append(task)

    # Spawn overflow pickers only for genuinely unplaceable tasks.
    overflow_picker_idx = num_pickers
    while still_unassigned:
        new_picker = _Picker(
            picker_id=f"{config.picker_id_prefix}-{overflow_picker_idx}"
        )
        overflow_picker_idx += 1
        batch_for_picker: List[PickTask] = []

        remaining: List[PickTask] = []
        for task in still_unassigned:
            if new_picker.capacity_used + task.qty <= max_units:
                new_picker.tasks.append(task)
                new_picker.capacity_used += task.qty
                new_picker.zones.add(task.zone)
                batch_for_picker.append(task)
            else:
                remaining.append(task)

        if batch_for_picker:
            pickers.append(new_picker)
            notes.append(
                f"overflow_picker={new_picker.picker_id} "
                f"tasks={len(batch_for_picker)} "
                f"capacity={new_picker.capacity_used}/{max_units}"
            )
        still_unassigned = remaining

    # ---- STEP 12–13: Reconstruct orders and finalise ----
    batches: List[List[Order]] = []
    batch_names: List[str] = []
    batch_types: List[str] = []

    for idx, picker in enumerate(pickers):
        if not picker.tasks:
            continue
        reconstructed = _reconstruct_orders(picker, original_orders)
        if not reconstructed:
            continue
        batches.append(reconstructed)
        batch_name = f"{config.batch_id_prefix}-{idx}"
        batch_names.append(batch_name)
        batch_type = "standard" if idx < num_pickers else "overflow"
        batch_types.append(batch_type)

        # Diagnostic notes.
        total_items = sum(order_unit_count(o) for o in reconstructed)
        unique_orders = sorted({o.order_id for o in reconstructed})
        route_dist = _task_route_distance(picker.tasks, grid, start, end)
        notes.append(
            f"Batch ID={batch_name} picker={picker.picker_id} "
            f"items={total_items}/{max_units} "
            f"zones={sorted(picker.zones)} "
            f"orders=[{','.join(unique_orders)}] "
            f"route_distance={route_dist:.2f}"
        )

    notes.append(
        f"total_pickers_used={len(batches)} "
        f"(configured={num_pickers}, overflow={max(0, len(batches) - num_pickers)})"
    )

    # STEP 13: Use finalize_assignment() for full compatibility.
    return finalize_assignment(batches, batch_names, batch_types, notes=notes)
