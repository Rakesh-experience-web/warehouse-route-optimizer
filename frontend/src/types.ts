export type Cell = { x: number; y: number };

export type ProductLocation = { sku: string; cell: Cell; category?: string };

export type OrderItem = { sku: string; qty: number };

export type Order = {
  order_id: string;
  items: OrderItem[];
  due_time_minutes: number;
  weight_score?: number;
};

export type OptimizationRequest = {
  layout: {
    width: number;
    height: number;
    blocked_cells: Cell[];
    shelf_cells: Cell[];
    path_cells: Cell[];
    depot: Cell;
    entry: Cell;
    exit: Cell;
  };
  product_map: ProductLocation[];
  orders: Order[];
  picker_speed_mps: number;
  config: {
    max_batch_size: number;
    max_batch_weight: number;
    batch_count: number;
    dynamic_batching_enabled: boolean;
    employee_count: number;
    max_shelf_visits_per_picker: number;
    similarity_batch_boost: number;
    allow_dynamic_reoptimization: boolean;
    alpha_distance: number;
    beta_due_time: number;
    gamma_weight: number;
    delta_similarity: number;
    use_ortools: boolean;
    allow_overflow_batches?: boolean;
  };
};

export type OptimizationResponse = {
  metrics: {
    naive_distance: number;
    batched_distance: number;
    improvement_pct: number;
    naive_time_seconds: number;
    batched_time_seconds: number;
    runtime_ms: number;
    prediction_eval_method?: string | null;
    prediction_r2?: number | null;
    prediction_mae?: number | null;
    prediction_rmse?: number | null;
    prediction_cv_r2_mean?: number | null;
    prediction_cv_mae_mean?: number | null;
    prediction_cv_rmse_mean?: number | null;
  };
  cluster_labels: Record<string, number>;
  notes: string[];
  overflow_batch_ids?: string[];
  unassigned_order_ids?: string[];
  batch_plans: {
    batch_id: string;
    order_ids: string[];
    picked_skus: string[];
    distance: number;
    estimated_seconds: number;
    route: { x: number; y: number; action: string; sku?: string }[];
  }[];
};

export type MapLayoutPayload = {
  name: string;
  layout: OptimizationRequest["layout"];
  shelf_categories: Record<string, string>;
};

export type StoredMap = {
  map_id: string;
  name: string;
  layout: OptimizationRequest["layout"];
  shelf_categories: Record<string, string>;
  created_at_epoch: number;
};
