"""workload_balancer.py — Workload and picker-load utilities.

These functions compute load-pressure metrics and determine which picker
should be assigned to a newly created batch.  They are kept separate from
scoring so that workload logic can be tested and tuned independently.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from app.optimizer.batching._types import BatchItem, SmartBatch
from app.optimizer.graph_model import GridGraph
from app.schemas import OptimizationConfig

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Load-pressure metrics
# ---------------------------------------------------------------------------

def picker_load_penalty(
    batch: SmartBatch,
    config: OptimizationConfig,
) -> float:
    """Return a normalised [0, 1] load score for *batch*.

    A value approaching 1.0 indicates the batch is near its size capacity.
    """
    soft_capacity = max(config.max_batch_size, 1)
    return batch.current_load / soft_capacity


# ---------------------------------------------------------------------------
# Picker assignment
# ---------------------------------------------------------------------------

def picker_id_for_new_batch(
    batches: List[SmartBatch],
    config: OptimizationConfig,
    start: Coord,
    first_item: BatchItem,
    grid: GridGraph,
) -> str:
    """Select the least-loaded picker for a newly created batch.

    Combines absolute load count with the travel cost from the depot to the
    first pick location so that idle pickers closest to the action are
    preferred over overloaded pickers that happen to be nearby.
    """
    picker_loads: Dict[str, int] = {
        f"{config.picker_id_prefix}-{idx}": 0
        for idx in range(max(config.employee_count, 1))
    }
    for batch in batches:
        picker_loads[batch.picker_id] = (
            picker_loads.get(batch.picker_id, 0) + batch.current_load
        )

    return min(
        picker_loads,
        key=lambda pid: grid.travel_cost(start, first_item.coord)
        + picker_loads[pid],
    )
