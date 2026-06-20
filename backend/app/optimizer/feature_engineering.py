from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from app.schemas import Order, ProductLocation

Coord = Tuple[int, int]
ProductLookupLike = Mapping[str, ProductLocation] | List[ProductLocation] | None


def build_product_lookup(product_map: List[ProductLocation]) -> Dict[str, ProductLocation]:
    return {p.sku: p for p in product_map}


def build_sku_lookup(product_map: List[ProductLocation]) -> Dict[str, Coord]:
    return {p.sku: (p.cell.x, p.cell.y) for p in product_map}


def build_sku_pick_node_lookup(
    product_map: List[ProductLocation],
    enable_pick_face: bool = True,
    walkable_nodes: Iterable[Coord] | None = None,
) -> Dict[str, Coord]:
    walkable = set(walkable_nodes) if walkable_nodes is not None else None
    out: Dict[str, Coord] = {}
    for p in product_map:
        fallback = (p.cell.x, p.cell.y)
        if enable_pick_face and p.pick_face is not None:
            pick_face = (p.pick_face.x, p.pick_face.y)
            if walkable is None or pick_face in walkable:
                out[p.sku] = pick_face
                continue
        out[p.sku] = fallback
    return out


def build_sku_category_lookup(product_map: List[ProductLocation]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in product_map:
        if p.category and p.category.strip():
            out[p.sku] = p.category.strip()
            continue
        # Fallback for SKU formats like "FRU-1", "DAI-7".
        out[p.sku] = p.sku.split("-", 1)[0]
    return out


def build_sku_zone_lookup(product_map: List[ProductLocation]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in product_map:
        if p.zone and p.zone.strip():
            out[p.sku] = p.zone.strip()
        elif p.category and p.category.strip():
            out[p.sku] = p.category.strip()
    return out


def order_pick_nodes(order: Order, sku_lookup: Dict[str, Coord]) -> List[Coord]:
    # Callers can now pass either raw storage-cell lookups or pick-face lookups.
    nodes: List[Coord] = []
    for item in order.items:
        coord = sku_lookup.get(item.sku)
        if coord:
            nodes.append(coord)
    return list(dict.fromkeys(nodes))


def order_centroid(order: Order, sku_lookup: Dict[str, Coord]) -> np.ndarray:
    coords = order_pick_nodes(order, sku_lookup)
    if not coords:
        return np.array([0.0, 0.0], dtype=np.float64)
    arr = np.array(coords, dtype=np.float64)
    return arr.mean(axis=0)


def _as_product_lookup(product_map_or_lookup: ProductLookupLike) -> Dict[str, ProductLocation]:
    if product_map_or_lookup is None:
        return {}
    if isinstance(product_map_or_lookup, Mapping):
        return dict(product_map_or_lookup)
    return build_product_lookup(list(product_map_or_lookup))


def order_weight(order: Order, product_map_or_lookup: ProductLookupLike = None) -> float:
    product_lookup = _as_product_lookup(product_map_or_lookup)
    total_weight = 0.0
    used_unit_weight = False
    for item in order.items:
        product = product_lookup.get(item.sku)
        if product and product.unit_weight is not None:
            total_weight += float(product.unit_weight) * item.qty
            used_unit_weight = True
    if used_unit_weight:
        return total_weight
    return float(sum(item.qty for item in order.items) * order.weight_score)


def order_volume(order: Order, product_map_or_lookup: ProductLookupLike = None) -> float:
    product_lookup = _as_product_lookup(product_map_or_lookup)
    total_volume = 0.0
    for item in order.items:
        product = product_lookup.get(item.sku)
        if product and product.unit_volume is not None:
            total_volume += float(product.unit_volume) * item.qty
    return total_volume


def order_unit_count(order: Order) -> int:
    return int(sum(item.qty for item in order.items))


def order_zone_set(order: Order, sku_zone_lookup: Dict[str, str]) -> set[str]:
    return {sku_zone_lookup[item.sku] for item in order.items if item.sku in sku_zone_lookup}


def order_fragility_flags(order: Order, product_map_or_lookup: ProductLookupLike = None) -> Dict[str, bool]:
    product_lookup = _as_product_lookup(product_map_or_lookup)
    return {
        "fragile": any(bool(product_lookup.get(item.sku).fragile) for item in order.items if item.sku in product_lookup),
        "bulky": any(bool(product_lookup.get(item.sku).bulky) for item in order.items if item.sku in product_lookup),
    }


def order_sku_counts(orders: List[Order]) -> Dict[str, int]:
    counts = defaultdict(int)
    for o in orders:
        for item in o.items:
            counts[item.sku] += item.qty
    return dict(counts)
