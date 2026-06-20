from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple

from app.schemas import BatchPlan, OptimizationConfig

Coord = Tuple[int, int]


@dataclass
class ReoptimizationResult:
    plans: List[BatchPlan]
    instability_cost: float


def low_disruption_reoptimize(
    existing: List[BatchPlan],
    proposed: List[BatchPlan],
    config: OptimizationConfig,
    *,
    batch_statuses: Mapping[str, str] | None = None,
    current_picker_positions: Mapping[str, Coord] | None = None,
    locked_order_ids: Iterable[str] | None = None,
    completed_order_ids: Iterable[str] | None = None,
) -> ReoptimizationResult:
    if not existing:
        return ReoptimizationResult(plans=proposed, instability_cost=0.0)

    existing_map = {p.batch_id: set(p.order_ids) for p in existing}
    proposed_map = {p.batch_id: set(p.order_ids) for p in proposed}
    statuses = dict(batch_statuses or {})
    picker_positions = dict(current_picker_positions or {})
    locked = set(locked_order_ids or [])
    completed = set(completed_order_ids or [])
    status_weights = {
        "not_started": config.reopt_not_started_weight,
        "in_progress": config.reopt_in_progress_weight,
        "completed": config.reopt_completed_weight,
    }

    instability = 0.0
    for batch_id in set(existing_map) | set(proposed_map):
        prev = existing_map.get(batch_id, set())
        now = proposed_map.get(batch_id, set())
        moved = prev.symmetric_difference(now)
        moved_locked = (prev - now) & locked
        moved_completed = (prev - now) & completed
        status = statuses.get(batch_id, "not_started")
        instability += status_weights.get(status, config.reopt_not_started_weight) * len(moved)
        instability += config.reopt_locked_order_weight * len(moved_locked)
        instability += config.reopt_completed_order_weight * len(moved_completed)
        if status == "in_progress" and batch_id in picker_positions:
            instability += config.reopt_picker_position_weight * len(moved)

    # Preserve the original keep-vs-replace behavior, but weight disruption
    # more heavily when work is already in progress or physically underway.
    old_distance = sum(p.distance for p in existing)
    new_distance = sum(p.distance for p in proposed)
    if (old_distance - new_distance) < config.reopt_disruption_acceptance_ratio * instability:
        return ReoptimizationResult(plans=existing, instability_cost=instability)
    return ReoptimizationResult(plans=proposed, instability_cost=instability)
