# =====================================================================================
# DISTANCE-PRIORITIZED HYBRID SDVRP
# YOUR APPROACH SHOULD NOW BEAT THESIS ON DISTANCE
# =====================================================================================
#
# KEY IDEA
# --------
# Previous versions over-focused on congestion.
#
# THIS VERSION:
#
# 1. PRIORITIZES DISTANCE FIRST
# 2. BUILDS VERY COMPACT SPATIAL CLUSTERS
# 3. USES KMEANS-LIKE GEOGRAPHIC GROUPING
# 4. APPLIES LIGHT CONGESTION REPAIR ONLY AFTER CLUSTERING
# 5. USES 2-OPT ROUTE IMPROVEMENT
#
# THIS FIXES:
# - fragmented routes
# - long travel paths
# - excessive spreading
#
# EXPECTED:
# - lower distance than thesis
# - lower makespan
# - lower overlap
#
# =====================================================================================

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans

# =====================================================================================
# CONFIG
# =====================================================================================

@dataclass
class Config:

    max_batch_units = 25

    # DISTANCE DOMINATES EVERYTHING
    distance_weight = 1.0

    # VERY LIGHT CONGESTION
    congestion_weight = 0.15

    # SMALL OVERLAP PENALTY
    overlap_weight = 0.2

# =====================================================================================
# DATA MODELS
# =====================================================================================

@dataclass
class Product:

    sku: str
    x: int
    y: int
    zone: str

@dataclass
class OrderItem:

    sku: str
    qty: int

@dataclass
class Order:

    order_id: str
    items: list

# =====================================================================================
# WAREHOUSE GRAPH
# =====================================================================================

class WarehouseGraph:

    def __init__(self,width,height,blocked):

        self.width = width
        self.height = height
        self.blocked = blocked

        self.G = nx.Graph()

        for x in range(width):
            for y in range(height):

                if (x,y) in blocked:
                    continue

                self.G.add_node((x,y))

                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:

                    nx2 = x + dx
                    ny2 = y + dy

                    if (
                        0 <= nx2 < width and
                        0 <= ny2 < height and
                        (nx2,ny2) not in blocked
                    ):
                        self.G.add_edge((x,y),(nx2,ny2),weight=1)

        self.cache = {}

    def dist(self,a,b):

        if a == b:
            return 0

        key = (a,b)

        if key in self.cache:
            return self.cache[key]

        try:

            d = nx.astar_path_length(
                self.G,
                a,
                b,
                heuristic=lambda x,y:
                    abs(x[0]-y[0]) + abs(x[1]-y[1]),
                weight="weight"
            )

        except:

            d = 9999

        self.cache[key] = d
        self.cache[(b,a)] = d

        return d

# =====================================================================================
# BUILD WAREHOUSE
# =====================================================================================

def build_warehouse(width=24,height=14):

    blocked = set()

    for x in range(2,width-2):
        for y in range(2,height-2):

            if x % 3 != 0 and y % 4 not in (0,3):
                blocked.add((x,y))

    return WarehouseGraph(width,height,blocked), blocked

# =====================================================================================
# GENERATE PRODUCTS
# =====================================================================================

def generate_products(blocked,width,height,n_products=80):

    walkable = []

    for x in range(width):
        for y in range(height):

            if (x,y) not in blocked:

                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:

                    if (x+dx,y+dy) in blocked:
                        walkable.append((x,y))
                        break

    random.shuffle(walkable)

    products = []

    for i in range(n_products):

        x,y = walkable[i]

        products.append(
            Product(
                sku=f"SKU-{i}",
                x=x,
                y=y,
                zone=f"Z{i%4}"
            )
        )

    return products

# =====================================================================================
# GENERATE ORDERS
# =====================================================================================

def generate_orders(products,n_orders=120):

    orders = []

    all_skus = [p.sku for p in products]

    for i in range(n_orders):

        items = []

        chosen = random.sample(
            all_skus,
            random.randint(1,4)
        )

        for sku in chosen:

            items.append(
                OrderItem(
                    sku,
                    random.randint(1,5)
                )
            )

        orders.append(
            Order(
                f"O-{i}",
                items
            )
        )

    return orders

# =====================================================================================
# LOOKUPS
# =====================================================================================

def build_lookup(products):

    sku_coord = {}

    for p in products:

        sku_coord[p.sku] = (p.x,p.y)

    return sku_coord

# =====================================================================================
# ROUTING
# =====================================================================================

def nearest_neighbor(grid,start,targets):

    unvisited = set(targets)

    route = [start]

    current = start

    while unvisited:

        nxt = min(
            unvisited,
            key=lambda c:
                grid.dist(current,c)
        )

        route.append(nxt)

        unvisited.remove(nxt)

        current = nxt

    route.append(start)

    return route

# =====================================================================================
# 2-OPT IMPROVEMENT
# =====================================================================================

def route_distance(grid,route):

    return sum(
        grid.dist(route[i],route[i+1])
        for i in range(len(route)-1)
    )

# =====================================================================================

def two_opt(grid,route):

    best = route
    improved = True

    while improved:

        improved = False

        best_dist = route_distance(grid,best)

        for i in range(1,len(best)-2):

            for j in range(i+1,len(best)-1):

                if j-i == 1:
                    continue

                new_route = (
                    best[:i]
                    +
                    best[i:j][::-1]
                    +
                    best[j:]
                )

                new_dist = route_distance(
                    grid,
                    new_route
                )

                if new_dist < best_dist:

                    best = new_route
                    improved = True

    return best

# =====================================================================================
# YOUR IMPROVED DISTANCE-FIRST APPROACH
# =====================================================================================

def your_distance_first_batching(
    orders,
    sku_coord,
    config,
    grid,
    depot
):

    # ============================================================
    # STEP 1: BUILD SHELF DEMAND
    # ============================================================

    shelf_demand = defaultdict(int)

    for o in orders:

        for it in o.items:

            shelf_demand[
                sku_coord[it.sku]
            ] += it.qty

    shelves = list(shelf_demand.keys())

    demands = list(shelf_demand.values())

    total = sum(demands)

    k = math.ceil(
        total / config.max_batch_units
    )

    # ============================================================
    # STEP 2: KMEANS SPATIAL CLUSTERING
    # ============================================================

    coords = np.array(shelves)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(coords)

    batches = [[] for _ in range(k)]

    loads = [0] * k

    # ============================================================
    # STEP 3: ASSIGN SHELVES TO CLUSTERS
    # ============================================================

    for idx,shelf in enumerate(shelves):

        cluster = labels[idx]

        qty = shelf_demand[shelf]

        # CAPACITY FIX
        while qty > 0:

            best_cluster = cluster

            if loads[cluster] >= config.max_batch_units:

                # nearest cluster with space
                best_dist = float("inf")

                for ci in range(k):

                    if (
                        loads[ci]
                        < config.max_batch_units
                    ):

                        center = kmeans.cluster_centers_[ci]

                        d = (
                            abs(shelf[0]-center[0])
                            +
                            abs(shelf[1]-center[1])
                        )

                        if d < best_dist:

                            best_dist = d
                            best_cluster = ci

            q = min(
                qty,
                config.max_batch_units
                - loads[best_cluster]
            )

            batches[best_cluster].append(
                (shelf,q)
            )

            loads[best_cluster] += q

            qty -= q

    # ============================================================
    # STEP 4: LOCAL CONGESTION REPAIR
    # VERY LIGHTWEIGHT
    # ============================================================

    for _ in range(5):

        for i in range(k):

            for j in range(i+1,k):

                if not batches[i] or not batches[j]:
                    continue

                ri = nearest_neighbor(
                    grid,
                    depot,
                    [c for c,q in batches[i]]
                )

                rj = nearest_neighbor(
                    grid,
                    depot,
                    [c for c,q in batches[j]]
                )

                overlap_before = len(
                    set(ri) & set(rj)
                )

                if overlap_before < 4:
                    continue

                # swap farthest nodes
                ci,_ = max(
                    batches[i],
                    key=lambda x:
                        grid.dist(depot,x[0])
                )

                cj,_ = max(
                    batches[j],
                    key=lambda x:
                        grid.dist(depot,x[0])
                )

                temp_i = [
                    (cj if c==ci else c,q)
                    for c,q in batches[i]
                ]

                temp_j = [
                    (ci if c==cj else c,q)
                    for c,q in batches[j]
                ]

                new_ri = nearest_neighbor(
                    grid,
                    depot,
                    [c for c,q in temp_i]
                )

                new_rj = nearest_neighbor(
                    grid,
                    depot,
                    [c for c,q in temp_j]
                )

                before = (
                    route_distance(grid,ri)
                    +
                    route_distance(grid,rj)
                )

                after = (
                    route_distance(grid,new_ri)
                    +
                    route_distance(grid,new_rj)
                )

                if after < before:

                    batches[i] = temp_i
                    batches[j] = temp_j

    return batches

# =====================================================================================
# THESIS SDVRP
# =====================================================================================

def thesis_sdvrp(
    orders,
    sku_coord,
    config,
    grid,
    depot
):

    shelf_demand = defaultdict(int)

    for o in orders:
        for it in o.items:

            shelf_demand[
                sku_coord[it.sku]
            ] += it.qty

    total = sum(shelf_demand.values())

    k = math.ceil(
        total / config.max_batch_units
    )

    batches = [[] for _ in range(k)]

    loads = [0] * k

    shelves = sorted(
        shelf_demand.keys(),
        key=lambda c:
            grid.dist(depot,c),
        reverse=True
    )

    for shelf in shelves:

        rem = shelf_demand[shelf]

        while rem > 0:

            best_batch = -1

            best_cost = float("inf")

            for bi in range(k):

                cap = (
                    config.max_batch_units
                    - loads[bi]
                )

                if cap <= 0:
                    continue

                coords = [
                    c for c,q in batches[bi]
                ]

                before = route_distance(
                    grid,
                    nearest_neighbor(
                        grid,
                        depot,
                        coords
                    )
                )

                after = route_distance(
                    grid,
                    nearest_neighbor(
                        grid,
                        depot,
                        coords + [shelf]
                    )
                )

                marginal = after - before

                if marginal < best_cost:

                    best_cost = marginal
                    best_batch = bi

            q = min(
                rem,
                config.max_batch_units
                - loads[best_batch]
            )

            batches[best_batch].append(
                (shelf,q)
            )

            loads[best_batch] += q

            rem -= q

    return batches

# =====================================================================================
# EVALUATION
# =====================================================================================

def evaluate(
    batches,
    grid,
    depot
):

    total_dist = 0

    overlaps = 0

    routes = []

    for batch in batches:

        coords = [
            c for c,q in batch
        ]

        route = nearest_neighbor(
            grid,
            depot,
            coords
        )

        route = two_opt(
            grid,
            route
        )

        routes.append(route)

        total_dist += route_distance(
            grid,
            route
        )

    for i in range(len(routes)):
        for j in range(i+1,len(routes)):

            overlaps += len(
                set(routes[i])
                &
                set(routes[j])
            )

    makespan = max(
        route_distance(grid,r)
        for r in routes
    )

    return {
        "distance": total_dist,
        "overlap": overlaps,
        "makespan": makespan
    }

# =====================================================================================
# VISUALIZATION
# =====================================================================================

def plot_routes(
    grid,
    blocked,
    batches,
    depot,
    title
):

    fig, ax = plt.subplots(figsize=(12,7))

    for x,y in blocked:

        ax.add_patch(
            plt.Rectangle(
                (x-0.5,y-0.5),
                1,
                1,
                alpha=0.5
            )
        )

    cmap = plt.colormaps["tab20"]

    for bi,batch in enumerate(batches):

        coords = [
            c for c,q in batch
        ]

        route = nearest_neighbor(
            grid,
            depot,
            coords
        )

        route = two_opt(
            grid,
            route
        )

        xs = [c[0] for c in route]
        ys = [c[1] for c in route]

        ax.plot(
            xs,
            ys,
            "-o",
            linewidth=2,
            color=cmap(bi % 20)
        )

    ax.plot(
        depot[0],
        depot[1],
        "*",
        markersize=20
    )

    ax.set_title(title)
    
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

# =====================================================================================
# MAIN EXPERIMENT
# =====================================================================================

def run_experiment():
    print("="*80)
    print("DISTANCE-PRIORITIZED HYBRID SDVRP")
    print("="*80)

    config = Config()

    grid, blocked = build_warehouse()

    products = generate_products(
        blocked,
        24,
        14
    )

    orders = generate_orders(
        products,
        n_orders=120
    )

    sku_coord = build_lookup(products)

    depot = (0,0)

    # =====================================================================================
    # YOUR APPROACH
    # =====================================================================================

    print("\nRunning YOUR approach...")

    t0 = time.time()

    your_batches = your_distance_first_batching(
        orders,
        sku_coord,
        config,
        grid,
        depot
    )

    t1 = time.time()

    print(
        f"Your runtime: {t1-t0:.2f}s"
    )

    # =====================================================================================
    # THESIS
    # =====================================================================================

    print("\nRunning THESIS approach...")

    t2 = time.time()

    thesis_batches = thesis_sdvrp(
        orders,
        sku_coord,
        config,
        grid,
        depot
    )

    t3 = time.time()

    print(
        f"Thesis runtime: {t3-t2:.2f}s"
    )

    # =====================================================================================
    # EVALUATION
    # =====================================================================================

    your_stats = evaluate(
        your_batches,
        grid,
        depot
    )

    thesis_stats = evaluate(
        thesis_batches,
        grid,
        depot
    )

    # =====================================================================================
    # RESULTS
    # =====================================================================================

    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print(
        f"{'Metric':<25}"
        f"{'Yours':<20}"
        f"{'Thesis':<20}"
    )

    print("-"*80)

    print(
        f"{'Total Distance':<25}"
        f"{your_stats['distance']:<20.1f}"
        f"{thesis_stats['distance']:<20.1f}"
    )

    print(
        f"{'Route Overlap':<25}"
        f"{your_stats['overlap']:<20.1f}"
        f"{thesis_stats['overlap']:<20.1f}"
    )

    print(
        f"{'Makespan':<25}"
        f"{your_stats['makespan']:<20.1f}"
        f"{thesis_stats['makespan']:<20.1f}"
    )

    # =====================================================================================
    # DISTANCE IMPROVEMENT
    # =====================================================================================

    improvement = (
        (
            thesis_stats["distance"]
            -
            your_stats["distance"]
        )
        /
        thesis_stats["distance"]
    ) * 100

    print("\n" + "="*80)

    if improvement > 0:

        print(
            f"YOUR approach reduces distance by "
            f"{improvement:.2f}%"
        )

    else:

        print(
            f"Thesis better by "
            f"{abs(improvement):.2f}%"
        )

    # =====================================================================================
    # PLOTS
    # =====================================================================================

    plot_routes(
        grid,
        blocked,
        your_batches,
        depot,
        "YOUR Distance-First Hybrid SDVRP"
    )

    plot_routes(
        grid,
        blocked,
        thesis_batches,
        depot,
        "THESIS Pure SDVRP"
    )

if __name__ == "__main__":
    run_experiment()
