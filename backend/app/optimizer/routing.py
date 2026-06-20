from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from app.optimizer.graph_model import GridGraph

Coord = Tuple[int, int]


def _distance_matrix(grid: GridGraph, nodes: Sequence[Coord]) -> List[List[int]]:
    matrix: List[List[int]] = []
    for i, src in enumerate(nodes):
        row = []
        for j, dst in enumerate(nodes):
            if i == j:
                row.append(0)
                continue
            d = grid.travel_cost(src, dst)
            row.append(int(math.ceil(d)))
        matrix.append(row)
    return matrix


def solve_route_ortools(grid: GridGraph, start: Coord, end: Coord, targets: Sequence[Coord]) -> List[Coord]:
    dedup_targets = [t for t in dict.fromkeys(targets) if t not in {start, end}]
    nodes = [start] + dedup_targets + [end]
    if len(nodes) == 2:
        return [start, end]
    if len(dedup_targets) <= 2:
        return solve_route_nearest_neighbor(grid, start, end, dedup_targets)

    dist = _distance_matrix(grid, nodes)
    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, [0], [len(nodes) - 1])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist[from_node][to_node]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        if len(dedup_targets) <= 6
        else routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # A fixed 2-second budget per batch is too expensive for interactive use.
    # Keep OR-Tools available, but scale its effort with route size.
    search_params.time_limit.FromMilliseconds(min(500, max(75, len(dedup_targets) * 40)))

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return solve_route_nearest_neighbor(grid, start, end, dedup_targets)

    order: List[Coord] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        order.append(nodes[node])
        idx = solution.Value(routing.NextVar(idx))
    order.append(end)
    return order


def solve_route_nearest_neighbor(grid: GridGraph, start: Coord, end: Coord, targets: Sequence[Coord]) -> List[Coord]:
    unvisited = set(t for t in targets if t not in {start, end})
    route = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda t: grid.travel_cost(current, t))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(end)
    return route


def solve_route_aisle_ordered(grid: GridGraph, start: Coord, end: Coord, targets: Sequence[Coord]) -> List[Coord]:
    dedup_targets = [t for t in dict.fromkeys(targets) if t not in {start, end}]
    if not dedup_targets:
        return [start, end]

    # This keeps the heuristic explainable for warehouse operators: visit aisle
    # groups in sorted order and snake within each aisle to reduce backtracking.
    unique_x = len({t[0] for t in dedup_targets})
    unique_y = len({t[1] for t in dedup_targets})
    aisle_axis = 0 if unique_x <= unique_y else 1
    cross_axis = 1 - aisle_axis
    grouped: Dict[int, List[Coord]] = defaultdict(list)
    for target in dedup_targets:
        grouped[target[aisle_axis]].append(target)

    ordered_aisles = sorted(grouped)
    route = [start]
    ascending = True
    for aisle in ordered_aisles:
        members = sorted(grouped[aisle], key=lambda coord: coord[cross_axis], reverse=not ascending)
        route.extend(members)
        ascending = not ascending
    route.append(end)
    return route


def solve_route_auto(
    grid: GridGraph,
    start: Coord,
    end: Coord,
    targets: Sequence[Coord],
    *,
    use_ortools: bool = True,
    prefer_warehouse_heuristic: bool = False,
) -> List[Coord]:
    dedup_targets = [t for t in dict.fromkeys(targets) if t not in {start, end}]
    if prefer_warehouse_heuristic and dedup_targets:
        return solve_route_aisle_ordered(grid, start, end, dedup_targets)
    if use_ortools:
        return solve_route_ortools(grid, start, end, dedup_targets)
    return solve_route_nearest_neighbor(grid, start, end, dedup_targets)


def route_distance(grid: GridGraph, ordered_nodes: Sequence[Coord]) -> float:
    distance = 0.0
    for i in range(len(ordered_nodes) - 1):
        distance += grid.travel_cost(ordered_nodes[i], ordered_nodes[i + 1])
    return distance


def expand_route_to_steps(grid: GridGraph, route_nodes: Sequence[Coord], sku_to_coord: Dict[str, Coord], cluster_order_ids: List[str]) -> List[dict]:
    coord_to_skus: Dict[Coord, List[str]] = {}
    for sku, c in sku_to_coord.items():
        coord_to_skus.setdefault(c, []).append(sku)

    steps: List[dict] = []
    if not route_nodes:
        return steps
    steps.append({"x": route_nodes[0][0], "y": route_nodes[0][1], "action": "start", "order_ids": cluster_order_ids})

    for i in range(len(route_nodes) - 1):
        src = route_nodes[i]
        dst = route_nodes[i + 1]
        path = grid.shortest_path(src, dst)
        for p in path[1:]:
            skus = coord_to_skus.get(p, [])
            if skus:
                steps.append(
                    {
                        "x": p[0],
                        "y": p[1],
                        "action": "pick",
                        "sku": skus[0],
                        "order_ids": cluster_order_ids,
                    }
                )
            else:
                steps.append({"x": p[0], "y": p[1], "action": "move", "order_ids": cluster_order_ids})
    steps[-1]["action"] = "end"
    return steps
