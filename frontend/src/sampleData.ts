import { OptimizationRequest } from "./types";

export function generateSamplePayload(orderCount = 40): OptimizationRequest {
  const width = 20;
  const height = 20;
  const skus = Array.from({ length: 60 }, (_, i) => `SKU-${i + 1}`);

  const product_map = skus.map((sku) => ({
    sku,
    cell: {
      x: 1 + Math.floor(Math.random() * (width - 2)),
      y: 1 + Math.floor(Math.random() * (height - 2))
    }
  }));

  const orders = Array.from({ length: orderCount }, (_, idx) => {
    const itemCount = 1 + Math.floor(Math.random() * 4);
    const items = Array.from({ length: itemCount }, () => ({
      sku: skus[Math.floor(Math.random() * skus.length)],
      qty: 1 + Math.floor(Math.random() * 2)
    }));
    return {
      order_id: `ORD-${idx + 1}`,
      items,
      due_time_minutes: 20 + Math.floor(Math.random() * 120),
      weight_score: 1 + Math.random()
    };
  });

  return {
    layout: {
      width,
      height,
      blocked_cells: [],
      shelf_cells: [],
      path_cells: [],
      depot: { x: 0, y: 0 },
      entry: { x: 0, y: 0 },
      exit: { x: width - 1, y: height - 1 }
    },
    product_map,
    orders,
    picker_speed_mps: 1.2,
    config: {
      max_batch_size: 8,
      max_batch_weight: 24,
      batch_count: 6,
      dynamic_batching_enabled: true,
      employee_count: 3,
      max_shelf_visits_per_picker: 18,
      similarity_batch_boost: 0.5,
      allow_dynamic_reoptimization: true,
      alpha_distance: 1,
      beta_due_time: 0.4,
      gamma_weight: 0.2,
      delta_similarity: 1.0,
      use_ortools: false
    }
  };
}
