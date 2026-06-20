"""similarity.py — SKU / category / zone similarity utilities.

All similarity functions operate on pure Python sets and return values in [0, 1].
No external dependencies beyond the standard library.
"""
from __future__ import annotations

from typing import Dict, List, Set

from app.schemas import Order


# ---------------------------------------------------------------------------
# Core Jaccard metric
# ---------------------------------------------------------------------------

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Return the Jaccard similarity coefficient between two string sets.

    Returns 1.0 when both sets are empty (treat identical empty sets as fully
    similar so that downstream cost functions do not penalise orders with no
    category information).
    """
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------

def average_pairwise_similarity(order_sku_sets: List[Set[str]]) -> float:
    """Compute the average Jaccard similarity over all (i, j) pairs.

    Used to quantify how homogeneous a set of orders is; high values indicate
    that splitting into fewer batches is likely to be beneficial.
    """
    n = len(order_sku_sets)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(order_sku_sets[i], order_sku_sets[j])
            pairs += 1
    return total / pairs if pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Order-level helpers
# ---------------------------------------------------------------------------

def order_category_set(order: Order, sku_category_lookup: Dict[str, str]) -> Set[str]:
    """Return the set of product categories referenced by this order's items.

    Falls back to the SKU prefix (e.g. ``"FRU"`` from ``"FRU-1"``) when no
    explicit category mapping is provided.
    """
    return {
        sku_category_lookup.get(item.sku, item.sku.split("-", 1)[0])
        for item in order.items
    }
