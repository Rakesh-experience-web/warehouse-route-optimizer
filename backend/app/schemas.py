from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Cell(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_coordinate(cls, value):
        # Backward-compatible convenience: allow coordinates to arrive as
        # `[x, y]` as well as `{ "x": x, "y": y }`.
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class DirectedEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_cell: Cell = Field(alias="from")
    to: Cell


class ProductLocation(BaseModel):
    sku: str
    cell: Cell
    category: Optional[str] = None
    pick_face: Optional[Cell] = None
    zone: Optional[str] = None
    fragile: bool = False
    bulky: bool = False
    unit_weight: Optional[float] = Field(default=None, ge=0.0)
    unit_volume: Optional[float] = Field(default=None, ge=0.0)


class WarehouseLayout(BaseModel):
    width: int = Field(gt=1)
    height: int = Field(gt=1)
    blocked_cells: List[Cell] = Field(default_factory=list)
    shelf_cells: List[Cell] = Field(default_factory=list)
    path_cells: List[Cell] = Field(default_factory=list)
    one_way_aisles: Dict[str, Literal["forward", "reverse"]] = Field(default_factory=dict)
    one_way_edges: List[DirectedEdge] = Field(default_factory=list)
    temporarily_blocked_cells: List[Cell] = Field(default_factory=list)
    turn_penalty: float = Field(default=0.0, ge=0.0)
    depot: Cell
    entry: Optional[Cell] = None
    exit: Optional[Cell] = None


class OrderItem(BaseModel):
    sku: str
    qty: int = Field(gt=0, le=99)


class Order(BaseModel):
    order_id: str
    items: List[OrderItem]
    due_time_minutes: int = Field(ge=1, le=1440, default=60)
    weight_score: float = Field(ge=0.0, default=1.0)
    created_at_epoch: int = Field(ge=0, default=0)
    priority: Optional[float] = None
    latest_pick_start_minutes: Optional[float] = Field(default=None, ge=0.0)
    temperature_sensitive: bool = False


class OptimizationConfig(BaseModel):
    max_batch_size: int = Field(gt=1, default=8)
    max_batch_weight: float = Field(gt=0, default=20.0)
    max_batch_volume: Optional[float] = Field(default=None, gt=0.0)
    max_batch_units: int = Field(default=25, gt=0)
    strict_category_grouping: bool = Field(default=True)
    max_batch_duration_seconds: Optional[float] = Field(default=None, gt=0.0)
    batch_count: int = Field(gt=0, default=4)
    dynamic_batching_enabled: bool = True
    employee_count: int = Field(gt=0, default=3)
    max_shelf_visits_per_picker: int = Field(gt=0, default=18)
    similarity_batch_boost: float = Field(ge=0, le=1, default=0.5)
    allow_dynamic_reoptimization: bool = True
    alpha_distance: float = Field(gt=0, default=1.0)
    beta_due_time: float = Field(ge=0, default=0.4)
    gamma_weight: float = Field(ge=0, default=0.2)
    delta_similarity: float = Field(ge=0, default=1.0)
    use_ortools: bool = True
    enable_pick_face_routing: bool = True
    use_insertion_batching: bool = True
    allow_overflow_batches: bool = True
    overflow_batch_name_prefix: str = "overflow"
    route_cost_reweight_factor: float = Field(default=1.0, ge=0.0)
    route_improvement_threshold: float = Field(default=1.0, ge=0.0)
    stable_new_batch_min_distance: float = Field(default=2.0, ge=0.0)
    stable_new_batch_distance_ratio: float = Field(default=0.5, ge=0.0)
    fragile_bulky_penalty: float = Field(default=0.4, ge=0.0)
    temperature_zone_mismatch_penalty: float = Field(default=0.2, ge=0.0)
    priority_score_weight: float = Field(default=0.05, ge=0.0)
    overflow_assignment_penalty: float = Field(default=0.15, ge=0.0)
    ils_iterations: int = Field(default=24, ge=0)
    ils_random_seed: int = Field(default=17, ge=0)
    ils_two_opt_passes: int = Field(default=2, ge=0)
    route_local_search_min_gain: float = Field(default=0.01, ge=0.0)
    min_capacity_denominator: float = Field(default=0.001, gt=0.0)
    picker_id_prefix: str = "picker"
    batch_id_prefix: str = "batch"
    advanced_aisle_penalty_weight: float = Field(default=1.0, ge=0.0)
    advanced_spread_penalty_weight: float = Field(default=0.25, ge=0.0)
    advanced_category_boost_weight: float = Field(default=0.35, ge=0.0)
    advanced_singleton_merge_max_delta: Optional[float] = Field(default=None, ge=0.0)
    zone_mismatch_penalty_weight: float = Field(default=50.0, ge=0.0)
    max_zones_per_batch: int = Field(default=2, ge=1)
    same_zone_nearby_boost_weight: float = Field(default=10.0, ge=0.0)
    dhobr_route_weight: float = Field(default=1.0, ge=0.0)
    dhobr_similarity_weight: float = Field(default=0.45, ge=0.0)
    dhobr_picker_load_weight: float = Field(default=0.25, ge=0.0)
    dhobr_delay_weight: float = Field(default=0.35, ge=0.0)
    dhobr_new_batch_bias: float = Field(default=0.0, ge=0.0)
    allow_order_splitting: bool = True
    split_improvement_threshold: float = Field(default=0.0, ge=0.0)
    reopt_not_started_weight: float = Field(default=1.0, ge=0.0)
    reopt_in_progress_weight: float = Field(default=2.0, ge=0.0)
    reopt_completed_weight: float = Field(default=3.0, ge=0.0)
    reopt_locked_order_weight: float = Field(default=3.0, ge=0.0)
    reopt_completed_order_weight: float = Field(default=4.0, ge=0.0)
    reopt_picker_position_weight: float = Field(default=0.5, ge=0.0)
    reopt_disruption_acceptance_ratio: float = Field(default=0.5, ge=0.0)
    advanced_route_weight: float = Field(default=1.0, ge=0.0)
    advanced_aisle_penalty_weight: float = Field(default=1.0, ge=0.0)
    advanced_spread_penalty_weight: float = Field(default=2.5, ge=0.0)
    advanced_category_boost_weight: float = Field(default=5.0, ge=0.0)


class CongestionReading(BaseModel):
    cell: Cell
    level: float = Field(ge=0.0)


class WarehouseTelemetry(BaseModel):
    global_congestion: Optional[float] = Field(default=None, ge=0.0)
    cell_congestion: List[CongestionReading] = Field(default_factory=list)
    zone_congestion: Dict[str, float] = Field(default_factory=dict)
    recorded_at_epoch: Optional[int] = Field(default=None, ge=0)


class ReoptimizationState(BaseModel):
    existing_batch_plans: List["BatchPlan"] = Field(default_factory=list)
    batch_statuses: Dict[str, Literal["not_started", "in_progress", "completed"]] = Field(default_factory=dict)
    current_picker_positions: Dict[str, Cell] = Field(default_factory=dict)
    locked_order_ids: List[str] = Field(default_factory=list)
    completed_order_ids: List[str] = Field(default_factory=list)


class OptimizationRequest(BaseModel):
    layout: WarehouseLayout
    product_map: List[ProductLocation]
    orders: List[Order]
    picker_speed_mps: float = Field(gt=0, default=1.2)
    config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    telemetry: Optional[WarehouseTelemetry] = None
    reoptimization: Optional[ReoptimizationState] = None


class RouteStep(BaseModel):
    x: int
    y: int
    action: Literal["start", "pick", "move", "end"]
    sku: Optional[str] = None
    order_ids: List[str] = Field(default_factory=list)


class BatchPlan(BaseModel):
    batch_id: str
    order_ids: List[str]
    picked_skus: List[str]
    distance: float
    estimated_seconds: float
    route: List[RouteStep]


class OptimizationMetrics(BaseModel):
    naive_distance: float
    batched_distance: float
    improvement_pct: float
    naive_time_seconds: float
    batched_time_seconds: float
    runtime_ms: float
    prediction_eval_method: Optional[str] = None
    prediction_r2: Optional[float] = None
    prediction_mae: Optional[float] = None
    prediction_rmse: Optional[float] = None
    prediction_cv_r2_mean: Optional[float] = None
    prediction_cv_mae_mean: Optional[float] = None
    prediction_cv_rmse_mean: Optional[float] = None


class OptimizationResponse(BaseModel):
    metrics: OptimizationMetrics
    batch_plans: List[BatchPlan]
    cluster_labels: Dict[str, int]
    notes: List[str] = Field(default_factory=list)
    overflow_batch_ids: List[str] = Field(default_factory=list)
    unassigned_order_ids: List[str] = Field(default_factory=list)


class TrainTravelTimeRequest(BaseModel):
    samples: List[Dict[str, float]]


class TrainTravelTimeResponse(BaseModel):
    model_path: str
    r2: float
    mae: float
    rmse: float
    evaluation_method: str
    cv_r2_mean: float
    cv_mae_mean: float
    cv_rmse_mean: float


class MapLayoutPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    layout: WarehouseLayout
    shelf_categories: Dict[str, str] = Field(default_factory=dict)


class StoredMap(BaseModel):
    map_id: str
    name: str
    layout: WarehouseLayout
    shelf_categories: Dict[str, str] = Field(default_factory=dict)
    created_at_epoch: int


class MapListResponse(BaseModel):
    maps: List[StoredMap]


ReoptimizationState.model_rebuild()
