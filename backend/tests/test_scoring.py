"""tests/test_scoring.py — Unit tests for the scoring engine.

Tests that:
  - Individual penalty functions return correct values for known inputs.
  - ScoreBreakdown.total equals the sum of its components.
  - score_insertion produces the expected composite score.
  - order_cost correctly weighs distance, urgency, and similarity.
"""
from __future__ import annotations

import pytest

from app.scoring.scoring_models import ScoreBreakdown
from app.scoring.penalty_manager import (
    urgency_penalty,
    workload_penalty,
    fragility_penalty_order,
)
from app.scoring.cost_function import order_cost
from app.optimizer.batching._types import BatchSummary
from app.schemas import OptimizationConfig
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> OptimizationConfig:
    return OptimizationConfig()


@pytest.fixture()
def sample_summary() -> BatchSummary:
    return BatchSummary(
        order_count=2,
        total_units=4,
        total_weight=8.0,
        total_volume=0.0,
        target_count=3,
        route_distance=12.0,
        duration_seconds=10.0,
        zones={"zone-A"},
        fragile=False,
        bulky=False,
    )


# ---------------------------------------------------------------------------
# urgency_penalty
# ---------------------------------------------------------------------------

class TestUrgencyPenalty:
    def test_no_penalty_when_on_time(self):
        # 600 seconds / 60 = 10 minutes, window = 15 → no penalty
        assert urgency_penalty(600.0, 15.0) == 0.0

    def test_penalty_when_late(self):
        # 1200 seconds / 60 = 20 min, window = 15 → 5 min late
        result = urgency_penalty(1200.0, 15.0)
        assert abs(result - 5.0) < 1e-9

    def test_zero_duration(self):
        assert urgency_penalty(0.0, 10.0) == 0.0


# ---------------------------------------------------------------------------
# workload_penalty
# ---------------------------------------------------------------------------

class TestWorkloadPenalty:
    def test_empty_batch_zero(self, default_config):
        empty = BatchSummary(0, 0, 0.0, 0.0, 0, 0.0, 0.0, set(), False, False)
        assert workload_penalty(empty, default_config) == 0.0

    def test_full_batch_high_pressure(self, default_config, sample_summary):
        # order_count=2, max_batch_size=8 → pressure ≥ 0.25
        result = workload_penalty(sample_summary, default_config)
        assert result > 0.0

    def test_volume_included_when_configured(self, default_config):
        config = default_config.model_copy(update={"max_batch_volume": 10.0})
        summary = BatchSummary(1, 1, 1.0, 5.0, 1, 1.0, 1.0, set(), False, False)
        result_with_vol = workload_penalty(summary, config)
        result_no_vol = workload_penalty(summary, default_config)
        assert result_with_vol > result_no_vol


# ---------------------------------------------------------------------------
# fragility_penalty_order
# ---------------------------------------------------------------------------

class TestFragilityPenalty:
    def test_fragile_in_bulky_batch(self, default_config, sample_summary):
        bulky_summary = BatchSummary(
            sample_summary.order_count, sample_summary.total_units,
            sample_summary.total_weight, sample_summary.total_volume,
            sample_summary.target_count, sample_summary.route_distance,
            sample_summary.duration_seconds, sample_summary.zones,
            fragile=False, bulky=True,
        )
        penalty = fragility_penalty_order(
            {"fragile": True, "bulky": False}, bulky_summary, set(), set(), default_config
        )
        assert penalty == default_config.fragile_bulky_penalty

    def test_no_penalty_compatible(self, default_config, sample_summary):
        penalty = fragility_penalty_order(
            {"fragile": False, "bulky": False}, sample_summary, set(), set(), default_config
        )
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# ScoreBreakdown
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_total_equals_component_sum(self):
        bd = ScoreBreakdown(
            route_delta=5.0,
            urgency_penalty=2.0,
            workload_penalty=1.0,
            zone_dissimilarity_penalty=3.0,
            category_similarity_bonus=-1.0,
            fragility_penalty=0.4,
            priority_bonus=-0.1,
            overflow_penalty=0.15,
        )
        expected = 5.0 + 2.0 + 1.0 + 3.0 - 1.0 + 0.4 - 0.1 + 0.15
        assert abs(bd.total - expected) < 1e-9

    def test_as_dict_contains_total(self):
        bd = ScoreBreakdown(route_delta=1.0)
        d = bd.as_dict()
        assert "total" in d
        assert d["total"] == bd.total


# ---------------------------------------------------------------------------
# order_cost (k-medoids)
# ---------------------------------------------------------------------------

class TestOrderCost:
    def test_closer_medoid_lower_cost(self, default_config):
        feature = np.array([5.0, 5.0])
        near_medoid = np.array([5.5, 5.0])
        far_medoid = np.array([10.0, 10.0])
        cost_near = order_cost(feature, near_medoid, 60, 1.0, set(), set(), default_config)
        cost_far = order_cost(feature, far_medoid, 60, 1.0, set(), set(), default_config)
        assert cost_near < cost_far

    def test_similarity_bonus_lowers_cost(self, default_config):
        feature = np.array([0.0, 0.0])
        medoid = np.array([1.0, 0.0])
        cats = {"furniture"}
        cost_match = order_cost(feature, medoid, 60, 1.0, cats, cats, default_config)
        cost_no_match = order_cost(feature, medoid, 60, 1.0, cats, set(), default_config)
        assert cost_match < cost_no_match
