from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

from app.ml.travel_time_model import TravelTimeEstimator
from app.schemas import Cell, OptimizationConfig, OptimizationRequest, Order, OrderItem, ProductLocation, WarehouseLayout
from app.services.optimizer_service import optimize_orders_with_strategy

Method = Literal["full", "greedy_nn", "spatial_ortools"]
Scenario = Literal["balanced", "hotspot", "urgent", "dynamic"]


@dataclass
class RunRecord:
    scenario: str
    method: str
    scale: int
    seed: int
    runtime_ms: float
    naive_distance: float
    batched_distance: float
    improvement_pct: float
    batched_time_seconds: float
    late_order_proxy: int
    dropped_pick_nodes: int
    total_pick_nodes: int
    dropped_pick_rate: float


@dataclass
class AblationRecord:
    scenario: str
    ablation: str
    scale: int
    seed: int
    runtime_ms: float
    batched_distance: float
    improvement_pct: float
    late_order_proxy: int


def _base_layout() -> WarehouseLayout:
    return WarehouseLayout(width=28, height=22, blocked_cells=[], shelf_cells=[], path_cells=[], depot=Cell(x=0, y=0))


def _catalog(rng: random.Random, sku_count: int = 120) -> List[ProductLocation]:
    return [
        ProductLocation(
            sku=f"SKU-{i + 1}",
            cell=Cell(x=rng.randint(1, 27), y=rng.randint(1, 21)),
        )
        for i in range(sku_count)
    ]


def _order_for_skus(rng: random.Random, order_id: str, skus: List[str], due_min: int, due_max: int, max_items: int = 4) -> Order:
    count = rng.randint(1, max_items)
    items = [OrderItem(sku=rng.choice(skus), qty=rng.randint(1, 2)) for _ in range(count)]
    return Order(
        order_id=order_id,
        items=items,
        due_time_minutes=rng.randint(due_min, due_max),
        weight_score=1 + rng.random(),
        created_at_epoch=int(time.time()),
    )


def _build_scenario_request(scenario: Scenario, scale: int, seed: int) -> OptimizationRequest:
    rng = random.Random(seed)
    layout = _base_layout()
    product_map = _catalog(rng)
    all_skus = [p.sku for p in product_map]
    hot_skus = all_skus[:20]

    orders: List[Order] = []
    for i in range(scale):
        oid = f"O-{i + 1}"
        if scenario == "hotspot":
            skus = hot_skus if rng.random() < 0.7 else all_skus
            orders.append(_order_for_skus(rng, oid, skus, due_min=20, due_max=180))
        elif scenario == "urgent":
            orders.append(_order_for_skus(rng, oid, all_skus, due_min=8, due_max=45))
        else:
            orders.append(_order_for_skus(rng, oid, all_skus, due_min=20, due_max=180))

    config = OptimizationConfig(
        batch_count=max(3, scale // 10),
        max_batch_size=10,
        max_batch_weight=35.0,
        allow_dynamic_reoptimization=scenario == "dynamic",
        alpha_distance=1.0,
        beta_due_time=0.45,
        gamma_weight=0.2,
        delta_similarity=1.25,
        use_ortools=True,
    )
    return OptimizationRequest(
        layout=layout,
        product_map=product_map,
        orders=orders,
        picker_speed_mps=1.2,
        config=config,
    )


def _summary(records: List[RunRecord]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[RunRecord]] = {}
    for r in records:
        key = f"{r.scenario}|{r.method}|N={r.scale}"
        groups.setdefault(key, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for key, vals in groups.items():
        out[key] = {
            "runs": float(len(vals)),
            "improvement_mean": statistics.mean(v.improvement_pct for v in vals),
            "improvement_std": statistics.pstdev(v.improvement_pct for v in vals) if len(vals) > 1 else 0.0,
            "runtime_ms_mean": statistics.mean(v.runtime_ms for v in vals),
            "late_proxy_mean": statistics.mean(v.late_order_proxy for v in vals),
            "dropped_rate_mean": statistics.mean(v.dropped_pick_rate for v in vals),
        }
    return out


def _ablation_summary(records: List[AblationRecord]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[AblationRecord]] = {}
    for r in records:
        key = f"{r.scenario}|{r.ablation}|N={r.scale}"
        groups.setdefault(key, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for key, vals in groups.items():
        out[key] = {
            "runs": float(len(vals)),
            "distance_mean": statistics.mean(v.batched_distance for v in vals),
            "improvement_mean": statistics.mean(v.improvement_pct for v in vals),
            "improvement_std": statistics.pstdev(v.improvement_pct for v in vals) if len(vals) > 1 else 0.0,
            "runtime_ms_mean": statistics.mean(v.runtime_ms for v in vals),
            "late_proxy_mean": statistics.mean(v.late_order_proxy for v in vals),
        }
    return out


def _write_csv(path: Path, rows: List[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must contain at least one record")
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def run_benchmark(
    output_dir: Path,
    scenarios: List[Scenario],
    methods: List[Method],
    scales: List[int],
    seeds: List[int],
) -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/travel_time_model.joblib")
    rows: List[RunRecord] = []

    for scenario in scenarios:
        for scale in scales:
            for seed in seeds:
                request = _build_scenario_request(scenario, scale, seed)
                existing_plans = None
                if scenario == "dynamic":
                    warm_req = _build_scenario_request("balanced", scale, seed + 10_000)
                    warm_res, _ = optimize_orders_with_strategy(warm_req, estimator=estimator, strategy="full")
                    existing_plans = warm_res.batch_plans
                for method in methods:
                    response, diagnostics = optimize_orders_with_strategy(
                        request,
                        estimator=estimator,
                        strategy=method,
                        existing_plans=existing_plans,
                    )
                    dropped_rate = (
                        diagnostics.dropped_pick_nodes / diagnostics.total_pick_nodes
                        if diagnostics.total_pick_nodes > 0
                        else 0.0
                    )
                    rows.append(
                        RunRecord(
                            scenario=scenario,
                            method=method,
                            scale=scale,
                            seed=seed,
                            runtime_ms=float(response.metrics.runtime_ms),
                            naive_distance=float(response.metrics.naive_distance),
                            batched_distance=float(response.metrics.batched_distance),
                            improvement_pct=float(response.metrics.improvement_pct),
                            batched_time_seconds=float(response.metrics.batched_time_seconds),
                            late_order_proxy=diagnostics.late_order_proxy,
                            dropped_pick_nodes=diagnostics.dropped_pick_nodes,
                            total_pick_nodes=diagnostics.total_pick_nodes,
                            dropped_pick_rate=dropped_rate,
                        )
                    )

    csv_path = output_dir / "benchmark_runs.csv"
    summary_path = output_dir / "benchmark_summary.json"
    _write_csv(csv_path, rows)
    summary = _summary(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} runs to {csv_path}")
    print(f"Wrote grouped summary to {summary_path}")


def run_ablations(output_dir: Path, scenarios: List[Scenario], scales: List[int], seeds: List[int]) -> None:
    estimator = TravelTimeEstimator(model_path="artifacts/travel_time_model.joblib")
    rows: List[AblationRecord] = []
    ablations = {
        "full": {},
        "no_due": {"beta_due_time": 0.0},
        "no_weight": {"gamma_weight": 0.0},
        "no_similarity": {"delta_similarity": 0.0},
        "no_reopt": {"allow_dynamic_reoptimization": False},
        "no_ortools": {"use_ortools": False},
    }

    for scenario in scenarios:
        for scale in scales:
            for seed in seeds:
                base_request = _build_scenario_request(scenario, scale, seed)
                for ablation_name, cfg_updates in ablations.items():
                    request = base_request.model_copy(
                        update={
                            "config": base_request.config.model_copy(update=cfg_updates),
                        }
                    )
                    response, diagnostics = optimize_orders_with_strategy(
                        request,
                        estimator=estimator,
                        strategy="full",
                        existing_plans=None,
                    )
                    rows.append(
                        AblationRecord(
                            scenario=scenario,
                            ablation=ablation_name,
                            scale=scale,
                            seed=seed,
                            runtime_ms=float(response.metrics.runtime_ms),
                            batched_distance=float(response.metrics.batched_distance),
                            improvement_pct=float(response.metrics.improvement_pct),
                            late_order_proxy=diagnostics.late_order_proxy,
                        )
                    )

    csv_path = output_dir / "ablation_runs.csv"
    summary_path = output_dir / "ablation_summary.json"
    _write_csv(csv_path, rows)
    summary = _ablation_summary(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} ablation runs to {csv_path}")
    print(f"Wrote ablation summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research benchmark harness for warehouse optimizer.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks", help="Output folder for CSV/JSON.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["balanced", "hotspot", "urgent", "dynamic"],
        choices=["balanced", "hotspot", "urgent", "dynamic"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "greedy_nn", "spatial_ortools"],
        choices=["full", "greedy_nn", "spatial_ortools"],
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-ablations", action="store_true", help="Skip ablation sweep.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        output_dir=Path(args.output_dir),
        scenarios=args.scenarios,
        methods=args.methods,
        scales=args.scales,
        seeds=args.seeds,
    )
    if not args.skip_ablations:
        run_ablations(
            output_dir=Path(args.output_dir),
            scenarios=args.scenarios,
            scales=args.scales,
            seeds=args.seeds,
        )
