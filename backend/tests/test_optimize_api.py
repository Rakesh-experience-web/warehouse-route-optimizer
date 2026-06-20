from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_optimize_smoke() -> None:
    payload = {
        "layout": {"width": 8, "height": 8, "blocked_cells": [], "depot": {"x": 0, "y": 0}},
        "product_map": [
            {"sku": "A", "cell": {"x": 1, "y": 1}},
            {"sku": "B", "cell": {"x": 6, "y": 6}},
            {"sku": "C", "cell": {"x": 2, "y": 6}},
        ],
        "orders": [
            {"order_id": "o1", "items": [{"sku": "A", "qty": 1}], "due_time_minutes": 30},
            {"order_id": "o2", "items": [{"sku": "B", "qty": 1}], "due_time_minutes": 60},
            {"order_id": "o3", "items": [{"sku": "C", "qty": 1}], "due_time_minutes": 45},
        ],
        "config": {"batch_count": 2, "max_batch_size": 3, "max_batch_weight": 20.0},
    }
    res = client.post("/api/v1/optimize", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "metrics" in body
    assert "batch_plans" in body


def test_optimize_accepts_execution_state_and_telemetry() -> None:
    payload = {
        "layout": {
            "width": 5,
            "height": 5,
            "blocked_cells": [],
            "path_cells": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}],
            "depot": {"x": 0, "y": 0},
            "entry": {"x": 0, "y": 0},
            "exit": {"x": 0, "y": 0},
        },
        "product_map": [
            {"sku": "A", "cell": {"x": 1, "y": 0}, "pick_face": {"x": 1, "y": 0}, "zone": "ambient"},
        ],
        "orders": [
            {"order_id": "o1", "items": [{"sku": "A", "qty": 1}], "due_time_minutes": 30},
        ],
        "telemetry": {
            "global_congestion": 0.1,
            "cell_congestion": [{"cell": {"x": 1, "y": 0}, "level": 0.3}],
            "zone_congestion": {"ambient": 0.2},
        },
        "reoptimization": {
            "existing_batch_plans": [
                {
                    "batch_id": "batch-0",
                    "order_ids": ["o1"],
                    "picked_skus": ["A"],
                    "distance": 2.0,
                    "estimated_seconds": 4.0,
                    "route": [{"x": 0, "y": 0, "action": "start", "order_ids": ["o1"]}],
                }
            ],
            "batch_statuses": {"batch-0": "in_progress"},
            "current_picker_positions": {"batch-0": {"x": 0, "y": 0}},
            "locked_order_ids": ["o1"],
            "completed_order_ids": [],
        },
        "config": {"batch_count": 1, "max_batch_size": 3, "max_batch_weight": 20.0, "use_ortools": False},
    }
    res = client.post("/api/v1/optimize", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "metrics" in body
    assert "overflow_batch_ids" in body
    assert "unassigned_order_ids" in body


def test_map_store_roundtrip() -> None:
    payload = {
        "name": "Test Map",
        "layout": {
            "width": 6,
            "height": 6,
            "blocked_cells": [],
            "shelf_cells": [{"x": 2, "y": 2}],
            "path_cells": [{"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 1}],
            "depot": {"x": 0, "y": 0},
            "entry": {"x": 0, "y": 0},
            "exit": {"x": 1, "y": 1},
        },
        "shelf_categories": {"2,2": "Fruits"},
    }
    save_res = client.post("/api/v1/maps", json=payload)
    assert save_res.status_code == 200
    saved = save_res.json()
    assert saved["name"] == "Test Map"
    map_id = saved["map_id"]

    list_res = client.get("/api/v1/maps")
    assert list_res.status_code == 200
    assert any(m["map_id"] == map_id for m in list_res.json()["maps"])

    get_res = client.get(f"/api/v1/maps/{map_id}")
    assert get_res.status_code == 200
    assert get_res.json()["map_id"] == map_id


def test_realistic_large_layout_with_many_orders() -> None:
    width = 18
    height = 14
    shelf_cells = [
        {"x": x, "y": y}
        for x in range(2, width - 2, 3)
        for y in range(2, height - 2)
        if y % 4 != 0
    ]
    products = [
        {"sku": f"SKU-{idx}", "cell": {"x": 1 + (idx * 5) % (width - 2), "y": 1 + (idx * 3) % (height - 2)}}
        for idx in range(20)
    ]
    shelf_set = {(cell["x"], cell["y"]) for cell in shelf_cells}
    for product in products:
        while (product["cell"]["x"], product["cell"]["y"]) in shelf_set:
            product["cell"]["x"] = (product["cell"]["x"] + 1) % width

    payload = {
        "layout": {
            "width": width,
            "height": height,
            "blocked_cells": [],
            "shelf_cells": shelf_cells,
            "depot": {"x": 0, "y": 0},
            "entry": {"x": 0, "y": 0},
            "exit": {"x": 0, "y": 0},
        },
        "product_map": products,
        "orders": [
            {
                "order_id": f"bulk-{idx}",
                "items": [{"sku": f"SKU-{idx % len(products)}", "qty": 1}],
                "due_time_minutes": 20 + (idx % 12),
                "created_at_epoch": idx * 2,
            }
            for idx in range(55)
        ],
        "config": {
            "batch_count": 8,
            "employee_count": 6,
            "max_batch_size": 9,
            "max_batch_weight": 50.0,
            "max_shelf_visits_per_picker": 12,
            "use_ortools": True,
            "route_improvement_threshold": 0.5,
        },
    }

    res = client.post("/api/v1/optimize", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body["batch_plans"]) > 0
    assert {order_id for batch in body["batch_plans"] for order_id in batch["order_ids"]} == {
        f"bulk-{idx}" for idx in range(55)
    }


def test_dense_obstacles_routes_never_cross_shelves() -> None:
    shelf_cells = [
        {"x": x, "y": y}
        for x in range(1, 8)
        for y in range(1, 8)
        if x not in {2, 5} and y not in {2, 5}
    ]
    payload = {
        "layout": {
            "width": 9,
            "height": 9,
            "blocked_cells": [],
            "shelf_cells": shelf_cells,
            "depot": {"x": 0, "y": 0},
            "entry": {"x": 0, "y": 0},
            "exit": {"x": 0, "y": 0},
        },
        "product_map": [
            {"sku": "A", "cell": {"x": 2, "y": 2}},
            {"sku": "B", "cell": {"x": 5, "y": 5}},
            {"sku": "C", "cell": {"x": 2, "y": 5}},
        ],
        "orders": [
            {"order_id": "dense-1", "items": [{"sku": "A", "qty": 1}], "due_time_minutes": 20},
            {"order_id": "dense-2", "items": [{"sku": "B", "qty": 1}], "due_time_minutes": 25},
            {"order_id": "dense-3", "items": [{"sku": "C", "qty": 1}], "due_time_minutes": 30},
        ],
        "config": {"batch_count": 2, "max_batch_size": 3, "max_batch_weight": 20.0},
    }

    res = client.post("/api/v1/optimize", json=payload)
    assert res.status_code == 200
    shelf_set = {(cell["x"], cell["y"]) for cell in shelf_cells}
    for batch in res.json()["batch_plans"]:
        for step in batch["route"]:
            assert (step["x"], step["y"]) not in shelf_set
