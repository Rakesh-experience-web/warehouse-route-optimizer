from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx

from app.schemas import Cell, WarehouseLayout


Coord = Tuple[int, int]
Direction = Coord
State = Tuple[Coord, Direction | None]


@dataclass(frozen=True)
class GridGraph:
    graph: nx.Graph | nx.DiGraph
    depot: Coord
    entry: Coord
    exit: Coord
    blocked: Set[Coord]
    width: int
    height: int
    turn_penalty: float = 0.0
    _shortest_path_length_cache: Dict[Tuple[Coord, Coord], float] = field(default_factory=dict, init=False, repr=False, compare=False)
    _shortest_path_cache: Dict[Tuple[Coord, Coord], List[Coord]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def shortest_path_length(self, src: Coord, dst: Coord) -> float:
        cached = self._shortest_path_length_cache.get((src, dst))
        if cached is not None:
            return cached
        if self.turn_penalty <= 0:
            path = nx.astar_path(self.graph, src, dst, heuristic=_manhattan, weight="weight")
            length = nx.path_weight(self.graph, path, weight="weight")
        else:
            length, path = self._turn_aware_shortest_path(src, dst)
        self._cache_path_result(src, dst, path, length)
        return length

    def shortest_path(self, src: Coord, dst: Coord) -> List[Coord]:
        cached = self._shortest_path_cache.get((src, dst))
        if cached is not None:
            return list(cached)
        if self.turn_penalty <= 0:
            path = nx.astar_path(self.graph, src, dst, heuristic=_manhattan, weight="weight")
            length = nx.path_weight(self.graph, path, weight="weight")
        else:
            length, path = self._turn_aware_shortest_path(src, dst)
        self._cache_path_result(src, dst, path, length)
        return list(path)

    def path_cost(self, path: List[Coord]) -> float:
        return path_cost(path, turn_penalty=self.turn_penalty)

    def travel_cost(self, src: Coord, dst: Coord) -> float:
        return self.shortest_path_length(src, dst)

    def _cache_path_result(self, src: Coord, dst: Coord, path: List[Coord], length: float) -> None:
        self._shortest_path_length_cache[(src, dst)] = length
        self._shortest_path_cache[(src, dst)] = list(path)
        if not self.graph.is_directed():
            self._shortest_path_length_cache[(dst, src)] = length
            self._shortest_path_cache[(dst, src)] = list(reversed(path))

    def _turn_aware_shortest_path(self, src: Coord, dst: Coord) -> Tuple[float, List[Coord]]:
        if src == dst:
            return 0.0, [src]
        if src not in self.graph or dst not in self.graph:
            raise nx.NodeNotFound(f"Either {src} or {dst} is not present in the warehouse graph.")

        start_state: State = (src, None)
        best_costs: Dict[State, float] = {start_state: 0.0}
        previous: Dict[State, State | None] = {start_state: None}
        heap: List[Tuple[float, Coord, Direction | None]] = [(0.0, src, None)]
        final_state: State | None = None

        while heap:
            cost, node, incoming_direction = heapq.heappop(heap)
            state = (node, incoming_direction)
            if cost > best_costs.get(state, float("inf")):
                continue
            if node == dst:
                final_state = state
                break

            for nxt in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, nxt) or {}
                base_cost = float(edge_data.get("base_weight", edge_data.get("weight", 1.0)))
                move_direction = _direction(node, nxt)
                turn_cost = self.turn_penalty if incoming_direction not in {None, move_direction} else 0.0
                next_cost = cost + base_cost + turn_cost
                next_state: State = (nxt, move_direction)
                if next_cost >= best_costs.get(next_state, float("inf")):
                    continue
                best_costs[next_state] = next_cost
                previous[next_state] = state
                heapq.heappush(heap, (next_cost, nxt, move_direction))

        if final_state is None:
            raise nx.NetworkXNoPath(f"No path between {src} and {dst}.")

        states: List[State] = []
        cursor: State | None = final_state
        while cursor is not None:
            states.append(cursor)
            cursor = previous[cursor]
        states.reverse()
        return best_costs[final_state], [coord for coord, _ in states]


def _cell_tuple(cell: Cell) -> Coord:
    return (cell.x, cell.y)


def _neighbors(x: int, y: int, width: int, height: int) -> Iterable[Coord]:
    if x > 0:
        yield (x - 1, y)
    if x < width - 1:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y < height - 1:
        yield (x, y + 1)


def _direction(src: Coord, dst: Coord) -> Coord:
    return (dst[0] - src[0], dst[1] - src[1])


def _manhattan(src: Coord, dst: Coord) -> float:
    return float(abs(src[0] - dst[0]) + abs(src[1] - dst[1]))


def edge_travel_cost(
    prev_node: Coord | None,
    current_node: Coord,
    next_node: Coord,
    base_cost: float = 1.0,
    turn_penalty: float = 0.0,
) -> float:
    if prev_node is None or turn_penalty <= 0:
        return base_cost
    return base_cost + (turn_penalty if _direction(prev_node, current_node) != _direction(current_node, next_node) else 0.0)


def path_cost(path: List[Coord], turn_penalty: float = 0.0) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(path)):
        prev_node = path[idx - 2] if idx > 1 else None
        total += edge_travel_cost(prev_node, path[idx - 1], path[idx], base_cost=1.0, turn_penalty=turn_penalty)
    return total


def _directed_override_set(layout: WarehouseLayout) -> Set[Tuple[Coord, Coord]]:
    overrides: Set[Tuple[Coord, Coord]] = set()
    for edge in layout.one_way_edges:
        src = _cell_tuple(edge.from_cell)
        dst = _cell_tuple(edge.to)
        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) != 1:
            continue
        overrides.add((src, dst))
    return overrides


def build_grid_graph(layout: WarehouseLayout) -> GridGraph:
    directed_overrides = _directed_override_set(layout)
    g = nx.DiGraph() if directed_overrides else nx.Graph()
    blocked = (
        {_cell_tuple(c) for c in layout.blocked_cells}
        | {_cell_tuple(c) for c in layout.shelf_cells}
        | {_cell_tuple(c) for c in layout.temporarily_blocked_cells}
    )
    allowed = {_cell_tuple(c) for c in layout.path_cells}
    has_allowed_filter = len(allowed) > 0

    for x in range(layout.width):
        for y in range(layout.height):
            c = (x, y)
            if c in blocked:
                continue
            if has_allowed_filter and c not in allowed:
                continue
            g.add_node(c)

    for x in range(layout.width):
        for y in range(layout.height):
            c = (x, y)
            if c in blocked or c not in g:
                continue
            for n in _neighbors(x, y, layout.width, layout.height):
                if n in blocked or n not in g:
                    continue
                if directed_overrides:
                    if (c, n) in directed_overrides:
                        g.add_edge(c, n, weight=1.0, base_weight=1.0)
                    elif (n, c) not in directed_overrides:
                        g.add_edge(c, n, weight=1.0, base_weight=1.0)
                else:
                    g.add_edge(c, n, weight=1.0, base_weight=1.0)

    depot = _cell_tuple(layout.depot)
    entry = _cell_tuple(layout.entry) if layout.entry else depot
    exit_node = _cell_tuple(layout.exit) if layout.exit else depot
    for label, node in [("Depot", depot), ("Entry", entry), ("Exit", exit_node)]:
        if node in blocked or node not in g:
            raise ValueError(f"{label} is blocked or outside walkable graph.")
    return GridGraph(
        graph=g,
        depot=depot,
        entry=entry,
        exit=exit_node,
        blocked=blocked,
        width=layout.width,
        height=layout.height,
        turn_penalty=layout.turn_penalty,
    )


def all_pairs_shortest_paths(grid: GridGraph, nodes: List[Coord]) -> Dict[Coord, Dict[Coord, float]]:
    unique_nodes = list(dict.fromkeys(nodes))
    out: Dict[Coord, Dict[Coord, float]] = {}
    for n in unique_nodes:
        out[n] = {}
        for other in unique_nodes:
            if n == other:
                out[n][other] = 0.0
                continue
            try:
                out[n][other] = grid.shortest_path_length(n, other)
            except nx.NetworkXNoPath:
                continue
    return out
