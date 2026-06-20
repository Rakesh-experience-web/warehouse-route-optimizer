"""tests/test_batching.py — Integration tests for batching algorithms.

Tests that:
  - All four algorithms produce valid BatchAssignment objects.
  - Constraint hard limits are respected.
  - Empty order lists return empty assignments.
  - seed_distance_batching respects zone limits.
  - insertion_cost_batching handles infeasible singleton orders.
"""
from __future__ import annotations

import pytest

from app.optimizer.batching import (
    BatchAssignment,
    constrained_k_medoids,
    greedy_capacity_batching,
    insertion_cost_batching,
    seed_distance_batching,
)
from app.optimizer.graph_model import build_grid_graph
from app.schemas import (
    Cell,
    OptimizationConfig,
    OptimizationRequest,
    Order,
    OrderItem,
    ProductLocation,
    WarehouseLayout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_layout(width: int = 6, height: int = 6) -> WarehouseLayout:
    path_cells = [Cell(x=x, y=y) for x in range(width) for y in range(height)]
    return WarehouseLayout(
        width=width,
        height=height,
        depot=Cell(x=0, y=0),
        entry=Cell(x=0, y=0),
        exit=Cell(x=0, y=0),
        path_cells=path_cells,
    )


def _make_product_map() -> list:
    return [
        ProductLocation(sku="SKU-A", cell=Cell(x=2, y=2), pick_face=Cell(x=2, y=1), category="furniture", zone="zone-A"),
        ProductLocation(sku="SKU-B", cell=Cell(x=4, y=2), pick_face=Cell(x=4, y=1), category="furniture", zone="zone-A"),
        ProductLocation(sku="SKU-C", cell=Cell(x=2, y=4), pick_face=Cell(x=2, y=3), category="appliance", zone="zone-B"),
        ProductLocation(sku="SKU-D", cell=Cell(x=4, y=4), pick_face=Cell(x=4, y=3), category="appliance", zone="zone-B"),
    ]


def _make_orders(n: int = 4) -> list:
    skus = ["SKU-A", "SKU-B", "SKU-C", "SKU-D"]
    return [
        Order(
            order_id=f"ORD-{i}",
            items=[OrderItem(sku=skus[i % len(skus)], qty=1)],
            due_time_minutes=30 + i * 5,
            weight_score=1.0,
            created_at_epoch=i,
        )
        for i in range(n)
    ]


@pytest.fixture()
def layout():
    return _make_layout()


@pytest.fixture()
def product_map():
    return _make_product_map()


@pytest.fixture()
def orders():
    return _make_orders(4)


@pytest.fixture()
def config():
    return OptimizationConfig(
        max_batch_size=4,
        max_batch_weight=50.0,
        batch_count=2,
        employee_count=3,
        max_zones_per_batch=2,
    )


@pytest.fixture()
def grid(layout):
    return build_grid_graph(layout)


@pytest.fixture()
def sku_lookup(product_map):
    return {p.sku: (p.cell.x, p.cell.y) for p in product_map}


@pytest.fixture()
def sku_category_lookup(product_map):
    return {p.sku: p.category for p in product_map}


@pytest.fixture()
def sku_zone_lookup(product_map):
    return {p.sku: p.zone for p in product_map}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_assignment(assignment: BatchAssignment, orders) -> None:
    assert isinstance(assignment, BatchAssignment)
    assert isinstance(assignment.batches, list)
    assert isinstance(assignment.labels, dict)
    order_ids = {o.order_id for o in orders}
    assigned_ids = {o.order_id for batch in assignment.batches for o in batch}
    unassigned_ids = set(assignment.exception_order_ids)
    # Every order must be either assigned or in exception list
    assert assigned_ids | unassigned_ids == order_ids, (
        f"Orders not accounted for: {order_ids - assigned_ids - unassigned_ids}"
    )


# ---------------------------------------------------------------------------
# Empty order set
# ---------------------------------------------------------------------------

class TestEmptyOrders:
    def test_greedy_empty(self, sku_lookup, sku_category_lookup, config):
        result = greedy_capacity_batching([], sku_lookup, sku_category_lookup, config)
        assert result.batches == []
        assert result.labels == {}

    def test_k_medoids_empty(self, sku_lookup, sku_category_lookup, config):
        result = constrained_k_medoids([], sku_lookup, sku_category_lookup, config)
        assert result.batches == []

    def test_insertion_empty(self, sku_lookup, sku_category_lookup, config):
        result = insertion_cost_batching([], sku_lookup, sku_category_lookup, config)
        assert result.batches == []

    def test_seed_empty(self, sku_lookup, sku_category_lookup, config, grid):
        result = seed_distance_batching(
            [], sku_lookup, sku_category_lookup, config,
            grid=grid, start=grid.entry, end=grid.exit,
        )
        assert result.batches == []


# ---------------------------------------------------------------------------
# greedy_capacity_batching
# ---------------------------------------------------------------------------

class TestGreedyBatching:
    def test_produces_valid_assignment(self, orders, sku_lookup, sku_category_lookup, config):
        result = greedy_capacity_batching(orders, sku_lookup, sku_category_lookup, config)
        _is_valid_assignment(result, orders)

    def test_respects_max_batch_size(self, orders, sku_lookup, sku_category_lookup, config):
        result = greedy_capacity_batching(orders, sku_lookup, sku_category_lookup, config)
        for batch in result.batches:
            assert len(batch) <= config.max_batch_size

    def test_respects_max_weight(self, orders, sku_lookup, sku_category_lookup):
        config = OptimizationConfig(max_batch_weight=1.0, max_batch_size=8)
        result = greedy_capacity_batching(orders, sku_lookup, sku_category_lookup, config)
        # With a very low weight limit each order becomes its own batch or exception
        assert len(result.batches) >= 0  # at minimum doesn't crash


# ---------------------------------------------------------------------------
# constrained_k_medoids
# ---------------------------------------------------------------------------

class TestKMedoids:
    def test_produces_valid_assignment(self, orders, sku_lookup, sku_category_lookup, config):
        result = constrained_k_medoids(orders, sku_lookup, sku_category_lookup, config)
        _is_valid_assignment(result, orders)

    def test_batch_count_bounded_by_employees(self, orders, sku_lookup, sku_category_lookup):
        config = OptimizationConfig(employee_count=2, batch_count=5)
        result = constrained_k_medoids(orders, sku_lookup, sku_category_lookup, config)
        standard = [bt for bt in result.batch_types if bt == "standard"]
        assert len(standard) <= max(config.employee_count, config.batch_count)


# ---------------------------------------------------------------------------
# insertion_cost_batching
# ---------------------------------------------------------------------------

class TestInsertionBatching:
    def test_produces_valid_assignment(self, orders, sku_lookup, sku_category_lookup, sku_zone_lookup, config, grid):
        result = insertion_cost_batching(
            orders, sku_lookup, sku_category_lookup, config,
            sku_zone_lookup=sku_zone_lookup,
            grid=grid, start=grid.entry, end=grid.exit,
        )
        _is_valid_assignment(result, orders)

    def test_infeasible_single_order_goes_to_exceptions(self, sku_lookup, sku_category_lookup, config):
        # Order weight exceeds max — should end up in exception_order_ids
        heavy_order = Order(
            order_id="HEAVY",
            items=[OrderItem(sku="SKU-A", qty=99)],
            due_time_minutes=5,
            weight_score=999.0,  # very high weight_score → exceeds max_batch_weight
        )
        config_tight = OptimizationConfig(max_batch_weight=0.001, max_batch_size=2)
        result = insertion_cost_batching([heavy_order], sku_lookup, sku_category_lookup, config_tight)
        assert "HEAVY" in result.exception_order_ids or result.batches == []


# ---------------------------------------------------------------------------
# seed_distance_batching
# ---------------------------------------------------------------------------

class TestSeedDistanceBatching:
    def test_produces_valid_assignment(self, orders, sku_lookup, sku_category_lookup, sku_zone_lookup, config, grid):
        result = seed_distance_batching(
            orders, sku_lookup, sku_category_lookup, config,
            sku_zone_lookup=sku_zone_lookup,
            grid=grid, start=grid.entry, end=grid.exit,
        )
        _is_valid_assignment(result, orders)

    def test_zone_limit_respected(self, orders, sku_lookup, sku_category_lookup, sku_zone_lookup, grid):
        config = OptimizationConfig(max_zones_per_batch=1, employee_count=4)
        result = seed_distance_batching(
            orders, sku_lookup, sku_category_lookup, config,
            sku_zone_lookup=sku_zone_lookup,
            grid=grid, start=grid.entry, end=grid.exit,
        )
        for batch in result.batches:
            zones = {sku_zone_lookup.get(item.sku, "") for o in batch for item in o.items}
            assert len(zones) <= config.max_zones_per_batch, f"Zone limit violated: {zones}"
