import networkx as nx
import pytest

from app.ml.travel_time_model import TravelTimeEstimator
from app.optimizer.graph_model import build_grid_graph
from app.schemas import Cell, OptimizationConfig, OptimizationRequest, Order, OrderItem, ProductLocation, WarehouseLayout
from app.services.optimizer_service import optimize_orders_with_strategy


def _request() -> OptimizationRequest:
    return OptimizationRequest(
        layout=WarehouseLayout(width=10, height=10, blocked_cells=[], shelf_cells=[], path_cells=[], depot=Cell(x=0, y=0)),
        product_map=[
            ProductLocation(sku="A", cell=Cell(x=1, y=1)),
            ProductLocation(sku="B", cell=Cell(x=4, y=4)),
            ProductLocation(sku="C", cell=Cell(x=7, y=2)),
            ProductLocation(sku="D", cell=Cell(x=3, y=8)),
        ],
        orders=[
            Order(order_id="o1", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1)], due_time_minutes=20),
            Order(order_id="o2", items=[OrderItem(sku="B", qty=1), OrderItem(sku="C", qty=1)], due_time_minutes=45),
            Order(order_id="o3", items=[OrderItem(sku="C", qty=1), OrderItem(sku="D", qty=1)], due_time_minutes=60),
            Order(order_id="o4", items=[OrderItem(sku="A", qty=1), OrderItem(sku="D", qty=1)], due_time_minutes=80),
        ],
        picker_speed_mps=1.2,
        config=OptimizationConfig(batch_count=2, max_batch_size=3, max_batch_weight=20.0, use_ortools=True, strict_category_grouping=False),
    )


def test_all_strategies_execute() -> None:
    req = _request()
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    for strategy in ["full", "greedy_nn", "spatial_ortools"]:
        res, diag = optimize_orders_with_strategy(req, estimator=estimator, strategy=strategy)
        assert len(res.batch_plans) > 0
        assert res.metrics.batched_distance >= 0
        assert res.metrics.naive_distance >= 0
        assert diag.strategy == strategy
        assert diag.total_pick_nodes >= 0


def test_dynamic_batching_responds_to_employee_capacity() -> None:
    req = _request()
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")

    low_staff = req.model_copy(
        deep=True,
        update={
            "config": req.config.model_copy(
                update={
                    "dynamic_batching_enabled": True,
                    "employee_count": 1,
                    "max_shelf_visits_per_picker": 1,
                    "max_batch_size": 10,
                    "max_batch_weight": 200.0,
                }
            )
        },
    )
    high_staff = req.model_copy(
        deep=True,
        update={
            "config": req.config.model_copy(
                update={
                    "dynamic_batching_enabled": True,
                    "employee_count": 4,
                    "max_shelf_visits_per_picker": 4,
                    "max_batch_size": 10,
                    "max_batch_weight": 200.0,
                }
            )
        },
    )

    low_res, _ = optimize_orders_with_strategy(low_staff, estimator=estimator, strategy="full")
    high_res, _ = optimize_orders_with_strategy(high_staff, estimator=estimator, strategy="full")
    assert len(low_res.batch_plans) <= len(high_res.batch_plans)
    assert len(low_res.batch_plans) <= low_staff.config.employee_count
    assert len(high_res.batch_plans) <= high_staff.config.employee_count


def test_order_splitting_respects_shelf_visit_limits_and_picker_cap() -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    req = OptimizationRequest(
        layout=WarehouseLayout(width=10, height=10, blocked_cells=[], shelf_cells=[], path_cells=[], depot=Cell(x=0, y=0)),
        product_map=[
            ProductLocation(sku="A", cell=Cell(x=1, y=1)),
            ProductLocation(sku="B", cell=Cell(x=2, y=2)),
            ProductLocation(sku="C", cell=Cell(x=3, y=3)),
        ],
        orders=[
            Order(order_id="o1", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1), OrderItem(sku="C", qty=1)], due_time_minutes=60),
        ],
        config=OptimizationConfig(
            batch_count=3,
            dynamic_batching_enabled=True,
            employee_count=3,
            max_batch_size=10,
            max_batch_weight=100.0,
            max_shelf_visits_per_picker=1,
            allow_order_splitting=True,
            use_ortools=False,
            strict_category_grouping=False,
        ),
    )

    res, _ = optimize_orders_with_strategy(req, estimator=estimator, strategy="split")
    assert len(res.batch_plans) <= req.config.employee_count
    assert len(res.batch_plans) == 3
    assert set(res.unassigned_order_ids) == set()
    assert [plan.order_ids for plan in res.batch_plans] == [["o1"], ["o1"], ["o1"]]


def test_dynamic_batching_similarity_extremes_use_1_to_n_pickers() -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    base = _request()

    very_similar = base.model_copy(
        deep=True,
        update={
            "orders": [
                Order(order_id="s1", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1)], due_time_minutes=20),
                Order(order_id="s2", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1)], due_time_minutes=30),
                Order(order_id="s3", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1)], due_time_minutes=40),
                Order(order_id="s4", items=[OrderItem(sku="A", qty=1), OrderItem(sku="B", qty=1)], due_time_minutes=50),
            ],
            "config": base.config.model_copy(
                update={
                    "dynamic_batching_enabled": True,
                    "employee_count": 4,
                    "max_batch_size": 10,
                    "max_batch_weight": 200.0,
                }
            ),
        },
    )
    dissimilar = base.model_copy(
        deep=True,
        update={
            "orders": [
                Order(order_id="d1", items=[OrderItem(sku="A", qty=1)], due_time_minutes=20),
                Order(order_id="d2", items=[OrderItem(sku="B", qty=1)], due_time_minutes=30),
                Order(order_id="d3", items=[OrderItem(sku="C", qty=1)], due_time_minutes=40),
                Order(order_id="d4", items=[OrderItem(sku="D", qty=1)], due_time_minutes=50),
            ],
            "config": base.config.model_copy(
                update={
                    "dynamic_batching_enabled": True,
                    "employee_count": 4,
                    "max_batch_size": 10,
                    "max_batch_weight": 200.0,
                }
            ),
        },
    )

    similar_res, _ = optimize_orders_with_strategy(very_similar, estimator=estimator, strategy="full")
    dissimilar_res, _ = optimize_orders_with_strategy(dissimilar, estimator=estimator, strategy="full")

    assert len(similar_res.batch_plans) == 1
    assert 1 <= len(dissimilar_res.batch_plans) <= 4
    assert {order_id for plan in dissimilar_res.batch_plans for order_id in plan.order_ids} == {
        "d1",
        "d2",
        "d3",
        "d4",
    }


def test_overflow_batches_capture_infeasible_extra_orders() -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    req = OptimizationRequest(
        layout=WarehouseLayout(width=6, height=6, blocked_cells=[], shelf_cells=[], path_cells=[], depot=Cell(x=0, y=0)),
        product_map=[
            ProductLocation(sku="A", cell=Cell(x=1, y=1)),
            ProductLocation(sku="B", cell=Cell(x=2, y=2)),
            ProductLocation(sku="C", cell=Cell(x=3, y=3)),
        ],
        orders=[
            Order(order_id="o1", items=[OrderItem(sku="A", qty=1)], due_time_minutes=20),
            Order(order_id="o2", items=[OrderItem(sku="B", qty=1)], due_time_minutes=30),
            Order(order_id="o3", items=[OrderItem(sku="C", qty=1)], due_time_minutes=40),
        ],
        config=OptimizationConfig(
            batch_count=1,
            dynamic_batching_enabled=False,
            employee_count=1,
            max_batch_size=2,
            max_batch_weight=100.0,
            max_shelf_visits_per_picker=10,
            allow_overflow_batches=True,
            use_ortools=False,
            strict_category_grouping=False,
        ),
    )

    res, _ = optimize_orders_with_strategy(req, estimator=estimator, strategy="full")

    assert len(res.overflow_batch_ids) == 1
    assert res.unassigned_order_ids == []
    assert sorted(order_id for plan in res.batch_plans for order_id in plan.order_ids) == ["o1", "o2", "o3"]


def test_directed_aisles_block_reverse_reachability() -> None:
    layout = WarehouseLayout(
        width=3,
        height=2,
        blocked_cells=[],
        shelf_cells=[],
        path_cells=[Cell(x=0, y=0), Cell(x=1, y=0), Cell(x=2, y=0)],
        one_way_edges=[
            {"from": [0, 0], "to": [1, 0]},
            {"from": [1, 0], "to": [2, 0]},
        ],
        depot=Cell(x=0, y=0),
        entry=Cell(x=0, y=0),
        exit=Cell(x=0, y=0),
    )
    grid = build_grid_graph(layout)

    assert nx.has_path(grid.graph, (0, 0), (2, 0))
    with pytest.raises(nx.NetworkXNoPath):
        grid.shortest_path((2, 0), (0, 0))


def test_pick_face_falls_back_to_storage_cell_when_not_walkable() -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    req = OptimizationRequest(
        layout=WarehouseLayout(
            width=4,
            height=3,
            blocked_cells=[],
            shelf_cells=[],
            path_cells=[Cell(x=0, y=0), Cell(x=1, y=0), Cell(x=2, y=0)],
            depot=Cell(x=0, y=0),
            entry=Cell(x=0, y=0),
            exit=Cell(x=0, y=0),
        ),
        product_map=[
            ProductLocation(
                sku="A",
                cell=Cell(x=1, y=0),
                pick_face=Cell(x=1, y=1),
            )
        ],
        orders=[Order(order_id="o1", items=[OrderItem(sku="A", qty=1)], due_time_minutes=30)],
        config=OptimizationConfig(use_ortools=False, enable_pick_face_routing=True, strict_category_grouping=False),
    )

    res, _ = optimize_orders_with_strategy(req, estimator=estimator, strategy="full")

    assert any(step.action == "pick" and step.x == 1 and step.y == 0 for step in res.batch_plans[0].route)


def test_zone_aware_batching_limits_zone_mixing() -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/nonexistent-model.joblib")
    req = OptimizationRequest(
        layout=WarehouseLayout(width=12, height=6, blocked_cells=[], shelf_cells=[], path_cells=[], depot=Cell(x=0, y=0)),
        product_map=[
            ProductLocation(sku="BAK-1", cell=Cell(x=1, y=1), category="Bakery", zone="Z1"),
            ProductLocation(sku="BAK-2", cell=Cell(x=2, y=1), category="Bakery", zone="Z1"),
            ProductLocation(sku="DAI-1", cell=Cell(x=8, y=1), category="Dairy", zone="Z2"),
            ProductLocation(sku="DAI-2", cell=Cell(x=9, y=1), category="Dairy", zone="Z2"),
            ProductLocation(sku="HH-1", cell=Cell(x=5, y=5), category="Household", zone="Z3"),
        ],
        orders=[
            Order(order_id="bak-1", items=[OrderItem(sku="BAK-1", qty=1)], due_time_minutes=30),
            Order(order_id="bak-2", items=[OrderItem(sku="BAK-2", qty=1)], due_time_minutes=30),
            Order(order_id="dai-1", items=[OrderItem(sku="DAI-1", qty=1)], due_time_minutes=30),
            Order(order_id="dai-2", items=[OrderItem(sku="DAI-2", qty=1)], due_time_minutes=30),
            Order(order_id="hh-1", items=[OrderItem(sku="HH-1", qty=1)], due_time_minutes=30),
        ],
        config=OptimizationConfig(
            batch_count=3,
            employee_count=3,
            max_batch_size=8,
            max_batch_weight=100.0,
            max_zones_per_batch=2,
            zone_mismatch_penalty_weight=50.0,
            use_ortools=False,
            strict_category_grouping=False,
        ),
    )

    res, _ = optimize_orders_with_strategy(req, estimator=estimator, strategy="full")
    sku_to_zone = {product.sku: product.zone for product in req.product_map}
    order_to_zone = {order.order_id: sku_to_zone[order.items[0].sku] for order in req.orders}

    assert all(len({order_to_zone[order_id] for order_id in plan.order_ids}) <= 2 for plan in res.batch_plans)
    assert any("Dominant zone=" in note for note in res.notes)
    bakery_batches = [plan.batch_id for plan in res.batch_plans if any(order_id.startswith("bak") for order_id in plan.order_ids)]
    assert len(set(bakery_batches)) == 1
