"""scoring_models.py — Score breakdown data structures.

Keeping the score breakdown as a typed dataclass makes debugging and tuning
much easier: callers can inspect individual component scores rather than
reasoning about a single opaque scalar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ScoreBreakdown:
    """Decomposed score for a candidate batch assignment.

    All component scores are in the same unit (lower is better).  Negative
    components represent bonuses (they reduce the total score).
    """

    # Route geometry
    route_delta: float = 0.0
    """Marginal increase in route distance caused by adding the order."""

    # Urgency
    urgency_penalty: float = 0.0
    """Penalty for projected late delivery."""

    # Capacity / workload
    workload_penalty: float = 0.0
    """Penalty for increasing load pressure on the batch."""

    # Zone similarity
    zone_dissimilarity_penalty: float = 0.0
    """Penalty for zone mismatch between the order and the batch."""

    # Category similarity (bonus → negative)
    category_similarity_bonus: float = 0.0
    """Reward for shared product categories (stored as negative value)."""

    # Fragility
    fragility_penalty: float = 0.0
    """Penalty for mixing fragile and bulky items."""

    # Priority (bonus → negative)
    priority_bonus: float = 0.0
    """Reward for high-priority orders (stored as negative value)."""

    # Overflow
    overflow_penalty: float = 0.0
    """Penalty for assigning to an overflow batch."""

    # Extra metadata
    extras: Dict[str, float] = field(default_factory=dict)
    """Algorithm-specific extra terms."""

    @property
    def total(self) -> float:
        """Sum of all component scores."""
        return (
            self.route_delta
            + self.urgency_penalty
            + self.workload_penalty
            + self.zone_dissimilarity_penalty
            + self.category_similarity_bonus
            + self.fragility_penalty
            + self.priority_bonus
            + self.overflow_penalty
            + sum(self.extras.values())
        )

    def as_dict(self) -> Dict[str, float]:
        """Return the breakdown as a flat dictionary for logging."""
        return {
            "route_delta": self.route_delta,
            "urgency_penalty": self.urgency_penalty,
            "workload_penalty": self.workload_penalty,
            "zone_dissimilarity_penalty": self.zone_dissimilarity_penalty,
            "category_similarity_bonus": self.category_similarity_bonus,
            "fragility_penalty": self.fragility_penalty,
            "priority_bonus": self.priority_bonus,
            "overflow_penalty": self.overflow_penalty,
            **self.extras,
            "total": self.total,
        }
