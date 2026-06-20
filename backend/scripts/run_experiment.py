from __future__ import annotations

import json
import random
import time

from app.ml.travel_time_model import TravelTimeEstimator
from app.schemas import Cell, OptimizationConfig, OptimizationRequest, Order, OrderItem, ProductLocation, WarehouseLayout
from app.services.optimizer_service import optimize_orders


def synthetic_request(order_count: int = 200) -> OptimizationRequest:
    width, height = 20, 20
    skus = [f"SKU-{i}" for i in range(1, 81)]
    product_map = [
        ProductLocation(sku=sku, cell=Cell(x=random.randint(1, width - 1), y=random.randint(1, height - 1)))
        for sku in skus
    ]
    orders = []
    for i in range(order_count):
        items = [
            OrderItem(sku=random.choice(skus), qty=random.randint(1, 2))
            for _ in range(random.randint(1, 4))
        ]
        orders.append(
            Order(
                order_id=f"O-{i+1}",
                items=items,
                due_time_minutes=random.randint(15, 180),
                weight_score=1 + random.random(),
                created_at_epoch=int(time.time()),
            )
        )
    return OptimizationRequest(
        layout=WarehouseLayout(width=width, height=height, blocked_cells=[], depot=Cell(x=0, y=0)),
        product_map=product_map,
        orders=orders,
        picker_speed_mps=1.2,
        config=OptimizationConfig(batch_count=10, max_batch_size=10, max_batch_weight=30.0),
    )


if __name__ == "__main__":
    req = synthetic_request()
    estimator = TravelTimeEstimator(model_path="artifacts/travel_time_model.joblib")
    res = optimize_orders(req, estimator=estimator)
    print(json.dumps(res.model_dump(), indent=2))

