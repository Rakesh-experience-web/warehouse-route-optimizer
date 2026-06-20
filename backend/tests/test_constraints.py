"""tests/test_constraints.py — Unit tests for the constraint engine."""
from __future__ import annotations

import pytest

from app.optimizer.batching._types import BatchSummary
from app.optimizer.batching.constraints import batch_constraint_violations, zone_limit_feasible
from app.schemas import OptimizationConfig, Order, OrderItem


def _summary(**overrides) -> BatchSummary:
    defaults = dict(
        order_count=2, total_units=4, total_weight=8.0, total_volume=0.0,
        target_count=3, route_distance=10.0, duration_seconds=30.0,
        zones={"zone-A"}, fragile=False, bulky=False,
    )
    defaults.update(overrides)
    return BatchSummary(**defaults)


@pytest.fixture()
def config():
    return OptimizationConfig()


class TestBatchConstraintViolations:
    def test_no_violations_for_valid_summary(self, config):
        summary = _summary()
        assert batch_constraint_violations(summary, config) == []

    def test_max_batch_size_violation(self, config):
        summary = _summary(order_count=config.max_batch_size + 1)
        violations = batch_constraint_violations(summary, config)
        assert "max_batch_size" in violations

    def test_max_batch_weight_violation(self, config):
        summary = _summary(total_weight=config.max_batch_weight + 1.0)
        violations = batch_constraint_violations(summary, config)
        assert "max_batch_weight" in violations

    def test_max_volume_violation_when_configured(self, config):
        config_vol = config.model_copy(update={"max_batch_volume": 5.0})
        summary = _summary(total_volume=6.0)
        violations = batch_constraint_violations(summary, config_vol)
        assert "max_batch_volume" in violations

    def test_max_volume_not_checked_when_none(self, config):
        summary = _summary(total_volume=9999.0)
        violations = batch_constraint_violations(summary, config)
        assert "max_batch_volume" not in violations

    def test_duration_violation_when_configured(self, config):
        config_dur = config.model_copy(update={"max_batch_duration_seconds": 10.0})
        summary = _summary(duration_seconds=60.0)
        violations = batch_constraint_violations(summary, config_dur)
        assert "max_batch_duration_seconds" in violations

    def test_shelf_visits_violation(self, config):
        summary = _summary(target_count=config.max_shelf_visits_per_picker + 1)
        violations = batch_constraint_violations(summary, config)
        assert "max_shelf_visits_per_picker" in violations

    def test_multiple_violations(self, config):
        summary = _summary(
            order_count=config.max_batch_size + 1,
            total_weight=config.max_batch_weight + 1.0,
        )
        violations = batch_constraint_violations(summary, config)
        assert len(violations) >= 2


class TestZoneLimitFeasible:
    def _make_order(self, order_id: str, sku: str) -> Order:
        return Order(order_id=order_id, items=[OrderItem(sku=sku, qty=1)], due_time_minutes=30)

    def test_single_zone_always_feasible(self, config):
        config_zone = config.model_copy(update={"max_zones_per_batch": 2})
        zone_lookup = {"SKU-A": "zone-A", "SKU-B": "zone-A"}
        batch = [self._make_order("O1", "SKU-A")]
        candidate = self._make_order("O2", "SKU-B")
        assert zone_limit_feasible(batch, candidate, zone_lookup, config_zone) is True

    def test_exceeding_zone_limit_returns_false(self, config):
        config_zone = config.model_copy(update={"max_zones_per_batch": 1})
        zone_lookup = {"SKU-A": "zone-A", "SKU-B": "zone-B"}
        batch = [self._make_order("O1", "SKU-A")]
        candidate = self._make_order("O2", "SKU-B")
        assert zone_limit_feasible(batch, candidate, zone_lookup, config_zone) is False
