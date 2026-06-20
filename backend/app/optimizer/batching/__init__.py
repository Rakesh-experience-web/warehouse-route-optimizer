"""optimizer/batching/__init__.py

Public re-export surface for the batching subpackage.

All callers that imported from ``app.optimizer.batching`` (the old monolith)
continue to work unchanged — this module re-exports every previously public
name so that backward compatibility is fully preserved.
"""
from __future__ import annotations

# Data types
from app.optimizer.batching._types import (
    BatchAssignment,
    BatchItem,
    BatchSummary,
    SmartBatch,
)

# Algorithms (public API)
from app.optimizer.batching.batch_builder import (
    constrained_k_medoids,
    global_zone_task_batching,
    greedy_capacity_batching,
    insertion_cost_batching,
    seed_distance_batching,
)

# Previously-public internal helpers still imported by incremental.py
from app.optimizer.batching._summary import (
    batch_targets as _batch_targets,
    summarize_batch as _summarize_batch,
)
from app.optimizer.batching.assignment import (
    effective_batch_count as _effective_batch_count,
    finalize_assignment as _finalize_assignment,
)
from app.optimizer.batching.constraints import (
    batch_constraint_violations as _batch_constraint_violations,
)

__all__ = [
    # Types
    "BatchAssignment",
    "BatchItem",
    "BatchSummary",
    "SmartBatch",
    # Algorithms
    "constrained_k_medoids",
    "global_zone_task_batching",
    "greedy_capacity_batching",
    "insertion_cost_batching",
    "seed_distance_batching",
    # Legacy internal names (imported by incremental.py)
    "_batch_targets",
    "_summarize_batch",
    "_effective_batch_count",
    "_finalize_assignment",
    "_batch_constraint_violations",
]
