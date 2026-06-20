"""utils/metrics.py — Structured optimisation metrics tracker.

Collects timing and quality metrics during an optimisation run and emits them
as a single structured log entry at the end.  This gives production dashboards
a consistent event schema to alert on.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetricsTracker:
    """Accumulate timing and quality metrics across one optimisation run.

    Usage::

        tracker = OptimizationMetricsTracker()
        with tracker.phase("batching"):
            assignment = seed_distance_batching(...)
        tracker.record("improvement_pct", 12.4)
        tracker.emit()
    """

    _phases: Dict[str, float] = field(default_factory=dict)
    _scalars: Dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    class _PhaseTimer:
        def __init__(self, tracker: "OptimizationMetricsTracker", name: str) -> None:
            self._tracker = tracker
            self._name = name
            self._t0: float = 0.0

        def __enter__(self) -> "_PhaseTimer":
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_: object) -> None:
            self._tracker._phases[self._name] = (time.perf_counter() - self._t0) * 1000

    def phase(self, name: str) -> "_PhaseTimer":
        """Context manager that records elapsed time for *name* in milliseconds."""
        return self._PhaseTimer(self, name)

    def record(self, key: str, value: float) -> None:
        """Store a scalar metric."""
        self._scalars[key] = value

    def total_ms(self) -> float:
        """Total elapsed milliseconds since this tracker was created."""
        return (time.perf_counter() - self._start) * 1000

    def emit(self, extra: Optional[Dict[str, object]] = None) -> None:
        """Emit all collected metrics as a single structured log entry."""
        payload: Dict[str, object] = {
            "total_ms": round(self.total_ms(), 2),
            **{f"phase_{k}_ms": round(v, 2) for k, v in self._phases.items()},
            **{k: round(v, 4) for k, v in self._scalars.items()},
            **(extra or {}),
        }
        logger.info("optimization_metrics %s", payload)

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_ms": round(self.total_ms(), 2),
            **{f"phase_{k}_ms": round(v, 2) for k, v in self._phases.items()},
            **self._scalars,
        }
