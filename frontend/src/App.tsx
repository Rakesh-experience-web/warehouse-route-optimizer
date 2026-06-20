import { useEffect, useMemo, useState } from "react";
import { getMap, listMaps, optimize, saveMap } from "./api";
import { Cell, OptimizationRequest, OptimizationResponse, Order, ProductLocation, StoredMap } from "./types";

type WorkflowPage = "landing" | "layout" | "orders" | "optimize" | "receipts";

type CatalogItem = {
  sku: string;
  name: string;
  category: string;
  price: number;
};

const CATEGORY_PRODUCTS: Record<string, string[]> = {
  Fruits: ["Apple", "Banana", "Mango", "Orange", "Grapes", "Pineapple", "Papaya", "Guava", "Kiwi", "Watermelon"],
  Vegetables: ["Tomato", "Potato", "Onion", "Carrot", "Cabbage", "Spinach", "Cauliflower", "Broccoli", "Capsicum", "Beetroot"],
  Dairy: ["Milk", "Curd", "Paneer", "Butter", "Cheese", "Ghee", "Yogurt", "Cream", "Buttermilk", "Lassi"],
  Bakery: ["Bread", "Bun", "Croissant", "Muffin", "Bagel", "Donut", "Pita", "Biscuit", "CakeSlice", "Rusk"],
  Beverages: ["Cola", "Juice", "GreenTea", "Coffee", "EnergyDrink", "SparklingWater", "Lemonade", "Malt", "Soda", "IcedTea"],
  Snacks: ["Chips", "Nachos", "Popcorn", "TrailMix", "GranolaBar", "Crackers", "Peanuts", "Cookies", "Pretzel", "Namkeen"],
  Frozen: ["FrozenPeas", "FrozenCorn", "IceCream", "FrozenPizza", "Nuggets", "Paratha", "Waffles", "Fries", "FishFillet", "Berries"],
  PersonalCare: ["Soap", "Shampoo", "Toothpaste", "Toothbrush", "Lotion", "FaceWash", "Deodorant", "Sanitizer", "Comb", "BodyWash"],
  Household: ["Detergent", "Dishwash", "FloorCleaner", "MopRefill", "TrashBag", "Tissue", "Foil", "Sponge", "ToiletCleaner", "GlassCleaner"],
  BabyCare: ["Diaper", "BabyWipes", "BabyLotion", "BabyPowder", "BabySoap", "BabyShampoo", "FeedingBottle", "Bib", "BabyFood", "Pacifier"]
};

function cardMetric(label: string, value: string) {
  return (
    <div className="metric-card" key={label}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function hashPrice(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const value = 30 + (h % 370);
  return Math.round(value * 100) / 100;
}

export function App() {
  const [page, setPage] = useState<WorkflowPage>("landing");
  const [width, setWidth] = useState<number>(24);
  const [height, setHeight] = useState<number>(20);
  const [orderCount, setOrderCount] = useState<number>(90);
  const [employeeCount, setEmployeeCount] = useState<number>(3);
  const [customSku, setCustomSku] = useState<string>("FRU-1");
  const [customQty, setCustomQty] = useState<number>(1);
  const [customDueTime, setCustomDueTime] = useState<number>(60);
  const [customWeightScore, setCustomWeightScore] = useState<number>(1);
  const [customItems, setCustomItems] = useState<Array<{ sku: string; qty: number }>>([]);
  const [mode, setMode] = useState<"shelf" | "path" | "entry" | "exit" | "clear">("shelf");
  const [activeCategory, setActiveCategory] = useState<string>("Fruits");
  const [entry, setEntry] = useState<Cell>({ x: 0, y: 0 });
  const [exitCell, setExitCell] = useState<Cell>({ x: 23, y: 19 });
  const [shelves, setShelves] = useState<Set<string>>(new Set());
  const [paths, setPaths] = useState<Set<string>>(new Set());
  const [shelfCategories, setShelfCategories] = useState<Record<string, string>>({});
  const [isPainting, setIsPainting] = useState<boolean>(false);
  const [orders, setOrders] = useState<Order[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<OptimizationResponse | null>(null);
  const [selectedBatch, setSelectedBatch] = useState<string | null>(null);
  const [mapName, setMapName] = useState<string>("Main Warehouse Map");
  const [savedMaps, setSavedMaps] = useState<StoredMap[]>([]);
  const [selectedMapId, setSelectedMapId] = useState<string>("");
  const [receiptPage, setReceiptPage] = useState<number>(1);
  const receiptsPerPage = 2;

  const categories = useMemo(() => Object.keys(CATEGORY_PRODUCTS), []);
  const catalog = useMemo<CatalogItem[]>(() => {
    const items: CatalogItem[] = [];
    categories.forEach((category) => {
      CATEGORY_PRODUCTS[category].forEach((name, idx) => {
        const sku = `${category.slice(0, 3).toUpperCase()}-${idx + 1}`;
        items.push({ sku, category, name, price: hashPrice(`${category}-${name}`) });
      });
    });
    return items;
  }, [categories]);
  const catalogBySku = useMemo(() => new Map(catalog.map((c) => [c.sku, c])), [catalog]);

  const cells = useMemo(() => {
    const out: Cell[] = [];
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) out.push({ x, y });
    }
    return out;
  }, [width, height]);

  function cellId(cell: Cell): string {
    return `${cell.x},${cell.y}`;
  }

  function parseCell(id: string): Cell {
    const [x, y] = id.split(",").map(Number);
    return { x, y };
  }

  function randomFrom<T>(arr: T[]): T {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function randomSample<T>(arr: T[], count: number): T[] {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, Math.max(0, Math.min(count, copy.length)));
  }

  function generateBills(): void {
    const skusByCategory = new Map<string, string[]>();
    catalog.forEach((c) => {
      const prev = skusByCategory.get(c.category) ?? [];
      prev.push(c.sku);
      skusByCategory.set(c.category, prev);
    });

    const generated: Order[] = [];
    for (let i = 0; i < orderCount; i += 1) {
      // Use partial basket coverage instead of "all categories in every order"
      // so higher-order simulations stay operationally realistic.
      const chosenCategories = randomSample(categories, 3 + Math.floor(Math.random() * 4));
      const items = chosenCategories.map((cat) => {
        const pool = skusByCategory.get(cat) ?? [];
        return { sku: randomFrom(pool), qty: 1 + Math.floor(Math.random() * 2) };
      });
      generated.push({
        order_id: `BILL-${i + 1}`,
        items,
        due_time_minutes: 20 + Math.floor(Math.random() * 90),
        weight_score: 1 + Math.random()
      });
    }
    setOrders(generated);
    setPage("orders");
  }

  function addCustomItem(): void {
    if (!customSku) return;
    setCustomItems((prev) => {
      const idx = prev.findIndex((i) => i.sku === customSku);
      if (idx === -1) return [...prev, { sku: customSku, qty: Math.max(1, customQty) }];
      const next = [...prev];
      next[idx] = { ...next[idx], qty: Math.max(1, next[idx].qty + customQty) };
      return next;
    });
  }

  function removeCustomItem(sku: string): void {
    setCustomItems((prev) => prev.filter((i) => i.sku !== sku));
  }

  function addCustomBill(): void {
    if (customItems.length === 0) {
      setError("Add at least one item to create a custom bill.");
      return;
    }
    setError("");
    const existingCustom = orders
      .map((o) => o.order_id)
      .filter((id) => id.startsWith("CUSTOM-"))
      .map((id) => Number(id.replace("CUSTOM-", "")))
      .filter((n) => Number.isFinite(n));
    const nextId = (existingCustom.length > 0 ? Math.max(...existingCustom) : 0) + 1;
    const newOrder: Order = {
      order_id: `CUSTOM-${nextId}`,
      items: customItems.map((i) => ({ sku: i.sku, qty: Math.max(1, i.qty) })),
      due_time_minutes: Math.max(1, customDueTime),
      weight_score: Math.max(0, customWeightScore)
    };
    setOrders((prev) => [...prev, newOrder]);
    setCustomItems([]);
    setCustomQty(1);
    setCustomDueTime(60);
    setCustomWeightScore(1);
    setPage("orders");
  }

  function buildProductMap(): ProductLocation[] {
    const allSkus = new Set<string>();
    orders.forEach((o) => o.items.forEach((i) => allSkus.add(i.sku)));
    const categoryBySku = new Map(catalog.map((c) => [c.sku, c.category]));

    const pathCells = Array.from(paths).map(parseCell);
    const hasPathMode = pathCells.length > 0;
    const shelfCellSet = new Set(shelves);

    function neighbors(c: Cell): Cell[] {
      const maybe = [
        { x: c.x + 1, y: c.y },
        { x: c.x - 1, y: c.y },
        { x: c.x, y: c.y + 1 },
        { x: c.x, y: c.y - 1 }
      ];
      return maybe.filter((n) => n.x >= 0 && n.y >= 0 && n.x < width && n.y < height);
    }

    const globalWalkable = cells.filter((c) => {
      const id = cellId(c);
      if (id === cellId(entry) || id === cellId(exitCell)) return true;
      if (shelfCellSet.has(id)) return false;
      return !hasPathMode || paths.has(id);
    });

    const categoryPickCells: Record<string, Cell[]> = {};
    categories.forEach((cat) => {
      const shelfCellsForCategory = Array.from(shelves)
        .filter((id) => shelfCategories[id] === cat)
        .map(parseCell);
      const around = shelfCellsForCategory.flatMap((s) =>
        neighbors(s).filter((n) => {
          const id = cellId(n);
          if (shelfCellSet.has(id)) return false;
          return !hasPathMode || paths.has(id);
        })
      );
      const uniqueById = new Map(around.map((c) => [cellId(c), c]));
      categoryPickCells[cat] = Array.from(uniqueById.values());
    });

    return Array.from(allSkus).map((sku, idx) => {
      const cat = categoryBySku.get(sku);
      const preferred = cat ? categoryPickCells[cat] : [];
      const pool = preferred.length > 0 ? preferred : globalWalkable;
      const safePool = pool.length > 0 ? pool : [entry];
      return { sku, cell: safePool[idx % safePool.length], category: cat };
    });
  }

  const currentLayout = useMemo(() => {
    const pathCells = Array.from(paths).map(parseCell);
    const pathSet = new Set(paths);
    pathSet.add(cellId(entry));
    pathSet.add(cellId(exitCell));
    return {
      width,
      height,
      blocked_cells: [],
      shelf_cells: Array.from(shelves).map(parseCell),
      path_cells: pathCells.length > 0 ? Array.from(pathSet).map(parseCell) : [],
      depot: entry,
      entry,
      exit: exitCell
    };
  }, [width, height, shelves, paths, entry, exitCell]);

  const payload: OptimizationRequest | null = useMemo(() => {
    if (orders.length === 0) return null;
    return {
      layout: currentLayout,
      product_map: buildProductMap(),
      orders,
      picker_speed_mps: 1.2,
      config: {
        max_batch_size: 12,
        max_batch_weight: 120,
        batch_count: Math.max(employeeCount, Math.ceil(orders.length / 6)),
        dynamic_batching_enabled: true,
        employee_count: employeeCount,
        max_shelf_visits_per_picker: 18,
        similarity_batch_boost: 0.5,
        allow_dynamic_reoptimization: true,
        alpha_distance: 1,
        beta_due_time: 0.45,
        gamma_weight: 0.2,
        delta_similarity: 1.25,
        use_ortools: false,
        allow_overflow_batches: true
      }
    };
  }, [orders, currentLayout, shelves, paths, shelfCategories, categories, catalog, employeeCount]);

  async function refreshMaps(): Promise<void> {
    const maps = await listMaps();
    setSavedMaps(maps);
    if (!selectedMapId && maps.length > 0) setSelectedMapId(maps[0].map_id);
  }

  async function onSaveMap(): Promise<void> {
    setError("");
    try {
      await saveMap({
        name: mapName.trim().length > 0 ? mapName.trim() : "Warehouse Map",
        layout: currentLayout,
        shelf_categories: shelfCategories
      });
      await refreshMaps();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Map save failed.");
    }
  }

  async function onLoadMap(): Promise<void> {
    if (!selectedMapId) return;
    setError("");
    try {
      const m = await getMap(selectedMapId);
      setMapName(m.name);
      setWidth(m.layout.width);
      setHeight(m.layout.height);
      setEntry(m.layout.entry ?? m.layout.depot);
      setExitCell(m.layout.exit ?? m.layout.depot);
      setShelves(new Set(m.layout.shelf_cells.map((c) => cellId(c))));
      setPaths(new Set(m.layout.path_cells.map((c) => cellId(c))));
      setShelfCategories(m.shelf_categories ?? {});
      setPage("layout");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Map load failed.");
    }
  }

  useEffect(() => {
    void refreshMaps();
  }, []);

  function paintCell(cell: Cell): void {
    const id = cellId(cell);
    if (mode === "entry") {
      setEntry(cell);
      setShelves((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setShelfCategories((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setPaths((prev) => new Set(prev).add(id));
      return;
    }
    if (mode === "exit") {
      setExitCell(cell);
      setShelves((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setShelfCategories((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setPaths((prev) => new Set(prev).add(id));
      return;
    }
    if (mode === "shelf") {
      setShelves((prev) => new Set(prev).add(id));
      setShelfCategories((prev) => ({ ...prev, [id]: activeCategory }));
      setPaths((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      return;
    }
    if (mode === "path") {
      setPaths((prev) => new Set(prev).add(id));
      setShelves((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setShelfCategories((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      return;
    }
    setShelves((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
    setShelfCategories((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setPaths((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  async function runOptimization(): Promise<void> {
    if (!payload) {
      setError("Generate order bills first.");
      setPage("orders");
      return;
    }
    setError("");
    setRunning(true);
    try {
      const response = await optimize(payload);
      setResult(response);
      if (response.batch_plans.length > 0) setSelectedBatch(response.batch_plans[0].batch_id);
      setPage("optimize");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Optimization failed.");
    } finally {
      setRunning(false);
    }
  }

  const selectedBatchPlan = result?.batch_plans.find((b) => b.batch_id === selectedBatch) ?? null;
  const evaluationStats = useMemo(() => {
    if (!result) return null;
    const totalOrdersAssigned = result.batch_plans.reduce((sum, batch) => sum + batch.order_ids.length, 0);
    const noteValue = (prefix: string): number | null => {
      const note = result.notes.find((entry) => entry.startsWith(prefix));
      if (!note) return null;
      const raw = note.slice(prefix.length);
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    };
    return {
      batchCount: result.batch_plans.length,
      overflowCount: result.overflow_batch_ids?.length ?? 0,
      unassignedCount: result.unassigned_order_ids?.length ?? 0,
      totalOrdersAssigned,
      avgOrdersPerBatch: result.batch_plans.length > 0 ? totalOrdersAssigned / result.batch_plans.length : 0,
      lateOrderProxy: noteValue("late_order_proxy="),
      droppedPickRate: noteValue("dropped_pick_rate="),
      naiveDroppedPickRate: noteValue("naive_dropped_pick_rate=")
    };
  }, [result]);
  const routeArrows = useMemo(() => {
    const out: Record<string, string> = {};
    if (!selectedBatchPlan) return out;
    for (let i = 0; i < selectedBatchPlan.route.length - 1; i += 1) {
      const a = selectedBatchPlan.route[i];
      const b = selectedBatchPlan.route[i + 1];
      const key = `${a.x},${a.y}`;
      if (b.x - a.x === 1) out[key] = ">";
      else if (b.x - a.x === -1) out[key] = "<";
      else if (b.y - a.y === 1) out[key] = "v";
      else if (b.y - a.y === -1) out[key] = "^";
    }
    return out;
  }, [selectedBatchPlan]);

  const orderMap = useMemo(() => new Map(orders.map((o) => [o.order_id, o])), [orders]);
  const receipts = useMemo(() => {
    if (!result) return [];
    return result.batch_plans.map((b) => {
      const qtyBySku = new Map<string, number>();
      const receiptRefsBySku = new Map<string, Map<string, number>>();
      b.order_ids.forEach((id) => {
        const ord = orderMap.get(id);
        if (!ord) return;
        ord.items.forEach((it) => {
          qtyBySku.set(it.sku, (qtyBySku.get(it.sku) ?? 0) + it.qty);
          const refs = receiptRefsBySku.get(it.sku) ?? new Map<string, number>();
          refs.set(id, (refs.get(id) ?? 0) + it.qty);
          receiptRefsBySku.set(it.sku, refs);
        });
      });
      const lines = Array.from(qtyBySku.entries())
        .map(([sku, qty]) => {
          const meta = catalogBySku.get(sku);
          const unitPrice = meta?.price ?? 0;
          const refs = Array.from((receiptRefsBySku.get(sku) ?? new Map()).entries())
            .map(([rid, q]) => `${rid}(${q})`)
            .join(", ");
          return {
            sku,
            name: meta?.name ?? sku,
            category: meta?.category ?? "Unknown",
            qty,
            unitPrice,
            total: Math.round(unitPrice * qty * 100) / 100,
            receiptRefs: refs
          };
        })
        .sort((a, z) => z.total - a.total);
      const grandTotal = Math.round(lines.reduce((sum, l) => sum + l.total, 0) * 100) / 100;
      return {
        batchId: b.batch_id,
        orders: b.order_ids.length,
        estSeconds: b.estimated_seconds,
        lines,
        grandTotal
      };
    });
  }, [result, orderMap, catalogBySku]);
  const totalReceiptPages = Math.max(1, Math.ceil(receipts.length / receiptsPerPage));
  const pagedReceipts = useMemo(
    () => receipts.slice((receiptPage - 1) * receiptsPerPage, receiptPage * receiptsPerPage),
    [receipts, receiptPage]
  );

  useEffect(() => {
    if (receiptPage > totalReceiptPages) setReceiptPage(totalReceiptPages);
  }, [receiptPage, totalReceiptPages]);

  useEffect(() => {
    setReceiptPage(1);
  }, [result]);

  const categoryCellCount = useMemo(() => {
    const out: Record<string, number> = {};
    categories.forEach((c) => (out[c] = 0));
    Object.values(shelfCategories).forEach((cat) => {
      out[cat] = (out[cat] ?? 0) + 1;
    });
    return out;
  }, [shelfCategories, categories]);

  return (
    <div className="app-shell" onMouseUp={() => setIsPainting(false)}>
      <div className="hero-bg" />
      <header className="topbar">
        <div>
          <h1>Warehouse Flow Studio</h1>
          <p>Deployable batch optimization UI with design, simulation, routing, and receipt intelligence.</p>
        </div>
        <nav className="page-nav">
          <button className={page === "landing" ? "active" : ""} onClick={() => setPage("landing")}>
            0. Landing
          </button>
          <button className={page === "layout" ? "active" : ""} onClick={() => setPage("layout")}>
            1. Layout
          </button>
          <button className={page === "orders" ? "active" : ""} onClick={() => setPage("orders")}>
            2. Orders
          </button>
          <button className={page === "optimize" ? "active" : ""} onClick={() => setPage("optimize")}>
            3. Optimize
          </button>
          <button className={page === "receipts" ? "active" : ""} onClick={() => setPage("receipts")}>
            4. Receipts
          </button>
        </nav>
      </header>
      {error && <div className="error-banner">{error}</div>}

      {page === "landing" && (
        <section className="page-screen">
          <section className="landing-card">
            <div className="landing-copy">
              <p className="kicker">Smart Grocery Warehouse Platform</p>
              <h2>From Shelf Design to Batch Packing in One Clean Flow</h2>
              <p>
                Create your map, generate category-balanced receipts, optimize picker routes, and pack using product-wise receipt
                references.
              </p>
              <div className="landing-actions">
                <button onClick={() => setPage("layout")}>Design Layout</button>
                <button className="ghost" onClick={() => setPage("orders")}>
                  Generate Bills
                </button>
              </div>
            </div>
            <div className="landing-stats">
              <div className="stat-box">
                <span>Categories</span>
                <strong>{categories.length}</strong>
              </div>
              <div className="stat-box">
                <span>Products</span>
                <strong>{catalog.length}</strong>
              </div>
              <div className="stat-box">
                <span>Saved Maps</span>
                <strong>{savedMaps.length}</strong>
              </div>
              <div className="stat-box">
                <span>Latest Batches</span>
                <strong>{result?.batch_plans.length ?? 0}</strong>
              </div>
            </div>
          </section>
        </section>
      )}

      {page === "layout" && (
        <section className="page-screen">
          <section className="panel">
          <h2>Layout Designer</h2>
          <div className="toolbar">
            <label>
              Width (ft)
              <input type="number" min={8} max={60} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
            </label>
            <label>
              Height (ft)
              <input type="number" min={8} max={60} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
            </label>
            <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
              <option value="shelf">Paint Shelves</option>
              <option value="path">Paint Walk Paths</option>
              <option value="entry">Set Entry Gate</option>
              <option value="exit">Set Exit Gate</option>
              <option value="clear">Clear Cells</option>
            </select>
            <select value={activeCategory} onChange={(e) => setActiveCategory(e.target.value)}>
              {categories.map((c) => (
                <option key={c} value={c}>
                  Shelf Category: {c}
                </option>
              ))}
            </select>
          </div>

          <div className="toolbar">
            <label>
              Map Name
              <input value={mapName} onChange={(e) => setMapName(e.target.value)} />
            </label>
            <button onClick={onSaveMap}>Save Map</button>
            <button onClick={refreshMaps}>Refresh</button>
            <select value={selectedMapId} onChange={(e) => setSelectedMapId(e.target.value)}>
              <option value="">Select Saved Map</option>
              {savedMaps.map((m) => (
                <option value={m.map_id} key={m.map_id}>
                  {m.name}
                </option>
              ))}
            </select>
            <button onClick={onLoadMap} disabled={!selectedMapId}>
              Load Map
            </button>
          </div>

          <p className="hint">Drag to paint continuous shelf arrays and path lanes. Entry and Exit must connect through paths.</p>
          <div className="grid-view" style={{ gridTemplateColumns: `repeat(${width}, 22px)` }}>
            {cells.map((c) => {
              const id = cellId(c);
              const cat = shelfCategories[id];
              const isEntry = id === cellId(entry);
              const isExit = id === cellId(exitCell);
              const isShelf = shelves.has(id);
              const isPath = paths.has(id);
              let cls = "empty";
              if (isShelf) cls = "shelf";
              if (isPath) cls = "path";
              if (isEntry) cls = "entry";
              if (isExit) cls = "exit";
              return (
                <button
                  key={id}
                  className={`cell ${cls}`}
                  onMouseDown={() => {
                    setIsPainting(true);
                    paintCell(c);
                  }}
                  onMouseEnter={() => {
                    if (isPainting) paintCell(c);
                  }}
                  title={`${cat ? `[${cat}] ` : ""}${id}`}
                  type="button"
                />
              );
            })}
          </div>
          <div className="chip-row">
            {categories.map((c) => (
              <div key={c} className={`chip ${activeCategory === c ? "active" : ""}`}>
                {c}: {categoryCellCount[c] ?? 0}
              </div>
            ))}
          </div>
          </section>
        </section>
      )}

      {page === "orders" && (
        <section className="page-screen">
          <section className="panel">
          <h2>Order Bills Generator</h2>
          <div className="toolbar">
            <label>
              Number of Bills
              <input type="number" min={20} max={800} value={orderCount} onChange={(e) => setOrderCount(Number(e.target.value))} />
            </label>
            <button onClick={generateBills}>Generate Bills</button>
            <button onClick={() => setOrders([])} disabled={orders.length === 0}>
              Clear Bills
            </button>
            <button onClick={() => setPage("optimize")} disabled={orders.length === 0}>
              Continue to Optimize
            </button>
          </div>
          <p className="hint">Each category includes 10 products. Bills are generated with 3-6 basket categories per order for more realistic higher-volume batching.</p>
          <h3>Custom Bill Input</h3>
          <div className="toolbar">
            <label>
              SKU
              <select value={customSku} onChange={(e) => setCustomSku(e.target.value)}>
                {catalog.map((item) => (
                  <option key={item.sku} value={item.sku}>
                    {item.sku} - {item.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Qty
              <input
                type="number"
                min={1}
                max={99}
                value={customQty}
                onChange={(e) => setCustomQty(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <button onClick={addCustomItem}>Add Item</button>
            <label>
              Due (min)
              <input
                type="number"
                min={1}
                max={1440}
                value={customDueTime}
                onChange={(e) => setCustomDueTime(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <label>
              Weight Score
              <input
                type="number"
                min={0}
                step={0.1}
                value={customWeightScore}
                onChange={(e) => setCustomWeightScore(Math.max(0, Number(e.target.value) || 0))}
              />
            </label>
            <button onClick={addCustomBill}>Create Custom Bill</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>SKU</th>
                  <th>Name</th>
                  <th>Qty</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {customItems.length === 0 ? (
                  <tr>
                    <td colSpan={4}>No items added yet.</td>
                  </tr>
                ) : (
                  customItems.map((item) => (
                    <tr key={item.sku}>
                      <td>{item.sku}</td>
                      <td>{catalogBySku.get(item.sku)?.name ?? item.sku}</td>
                      <td>{item.qty}</td>
                      <td>
                        <button type="button" onClick={() => removeCustomItem(item.sku)}>
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <div className="metrics-grid">
            {cardMetric("Categories", String(categories.length))}
            {cardMetric("Products", String(catalog.length))}
            {cardMetric("Generated Bills", String(orders.length))}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Bill ID</th>
                  <th>Category Coverage</th>
                  <th>Sample SKUs</th>
                </tr>
              </thead>
              <tbody>
                {orders.slice(0, 14).map((o) => (
                  <tr key={o.order_id}>
                    <td>{o.order_id}</td>
                    <td>{o.items.length}/{categories.length}</td>
                    <td>{o.items.slice(0, 3).map((i) => i.sku).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          </section>
        </section>
      )}

      {page === "optimize" && (
        <section className="page-screen">
          <section className="panel">
          <h2>Batch Optimization</h2>
          <div className="toolbar">
            <button onClick={runOptimization} disabled={running || orders.length === 0}>
              {running ? "Optimizing..." : "Run Optimization"}
            </button>
            <label>
              Pickers Available
              <input
                type="number"
                min={1}
                max={50}
                value={employeeCount}
                onChange={(e) => setEmployeeCount(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <button onClick={() => setPage("receipts")} disabled={!result}>
              View Batch Receipts
            </button>
          </div>
          {result && (
            <>
              <div className="metrics-grid">
                {cardMetric("Naive Distance", result.metrics.naive_distance.toFixed(2))}
                {cardMetric("Batched Distance", result.metrics.batched_distance.toFixed(2))}
                {cardMetric("Improvement", `${result.metrics.improvement_pct.toFixed(2)}%`)}
                {cardMetric("Runtime", `${result.metrics.runtime_ms.toFixed(1)} ms`)}
                {cardMetric("Overflow Batches", String(result.overflow_batch_ids?.length ?? 0))}
                {cardMetric("Unassigned Orders", String(result.unassigned_order_ids?.length ?? 0))}
              </div>

              <div className="batch-grid">
                {result.batch_plans.map((b) => (
                  <article
                    key={b.batch_id}
                    className={`batch-card ${selectedBatch === b.batch_id ? "selected" : ""}`}
                    onClick={() => setSelectedBatch(b.batch_id)}
                  >
                    <h3>{b.batch_id}</h3>
                    <p>Orders: {b.order_ids.length}</p>
                    <p>SKUs: {b.picked_skus.length}</p>
                    <p>Distance: {b.distance.toFixed(2)}</p>
                    <p>ETA: {b.estimated_seconds.toFixed(1)}s</p>
                  </article>
                ))}
              </div>

              {selectedBatchPlan && (
                <div>
                  <h3>{selectedBatchPlan.batch_id} Route Directions</h3>
                  <div className="grid-view" style={{ gridTemplateColumns: `repeat(${width}, 22px)` }}>
                    {cells.map((c) => {
                      const id = cellId(c);
                      const marker = selectedBatchPlan.route.find((r) => r.x === c.x && r.y === c.y);
                      const isEntry = id === cellId(entry);
                      const isExit = id === cellId(exitCell);
                      const isShelf = shelves.has(id);
                      let cls = isShelf ? "shelf" : "empty";
                      if (marker?.action === "move") cls = "route";
                      if (marker?.action === "pick") cls = "pick";
                      if (isEntry) cls = "entry";
                      if (isExit) cls = "exit";
                      return (
                        <div className={`cell ${cls}`} key={`r-${id}`} title={marker?.sku ?? id}>
                          <span className="arrow">{routeArrows[id] ?? ""}</span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="hint">Arrow marks show direction of movement for this batch route.</p>
                </div>
              )}
            </>
          )}
          </section>
        </section>
      )}

      {page === "receipts" && (
        <section className="page-screen">
          <section className="panel">
          <h2>Batch Receipts</h2>
          {!result && <p className="hint">Run optimization first to generate receipts.</p>}
          {result && (
            <>
              <div className="receipt-grid">
              {pagedReceipts.map((r) => (
                <article key={r.batchId} className="receipt-card">
                  <div className="receipt-head">
                    <div>
                      <h3>{r.batchId}</h3>
                      <p>Orders: {r.orders}</p>
                    </div>
                    <div>
                      <div className="receipt-total">Rs. {r.grandTotal.toFixed(2)}</div>
                      <p>ETA: {r.estSeconds.toFixed(1)}s</p>
                    </div>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Product</th>
                          <th>Cat</th>
                          <th>Qty</th>
                          <th>Unit</th>
                          <th>Total</th>
                          <th>Receipt IDs</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.lines.slice(0, 14).map((line) => (
                          <tr key={`${r.batchId}-${line.sku}`}>
                            <td>{line.name}</td>
                            <td>{line.category}</td>
                            <td>{line.qty}</td>
                            <td>{line.unitPrice.toFixed(2)}</td>
                            <td>{line.total.toFixed(2)}</td>
                            <td>{line.receiptRefs}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              ))}
              </div>
              <div className="pagination-bar">
                <button onClick={() => setReceiptPage((p) => Math.max(1, p - 1))} disabled={receiptPage <= 1}>
                  Prev
                </button>
                <span>
                  Page {receiptPage} / {totalReceiptPages}
                </span>
                <button
                  onClick={() => setReceiptPage((p) => Math.min(totalReceiptPages, p + 1))}
                  disabled={receiptPage >= totalReceiptPages}
                >
                  Next
                </button>
              </div>
            </>
          )}
          </section>
        </section>
      )}
    </div>
  );
}
