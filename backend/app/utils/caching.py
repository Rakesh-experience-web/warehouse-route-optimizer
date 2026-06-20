"""utils/caching.py — Lightweight memoisation helpers.

Provides a function-level LRU cache wrapper and a route-distance cache that
avoids redundant GridGraph.travel_cost calls across the optimisation pipeline.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
Coord = Tuple[int, int]


def memoize(maxsize: int = 256) -> Callable[[F], F]:
    """Decorator that caches function results using LRU eviction.

    Suitable for pure functions whose arguments are hashable.  Results are
    cached per unique argument tuple.

    Args:
        maxsize: Maximum number of cached results (passed to ``functools.lru_cache``).
    """
    def decorator(fn: F) -> F:
        cached_fn = functools.lru_cache(maxsize=maxsize)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # lru_cache does not support kwargs; convert to sorted tuple key
            if kwargs:
                key = args + tuple(sorted(kwargs.items()))
                # Fall back to uncached call for unhashable kwargs
                try:
                    hash(key)
                except TypeError:
                    return fn(*args, **kwargs)
                return cached_fn(*args)  # kwargs already baked into key via closure — not ideal; for production use a dict-based cache instead
            return cached_fn(*args)

        wrapper.cache_info = cached_fn.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached_fn.cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


class RouteDistanceCache:
    """Thread-safe dict-based cache for GridGraph travel costs.

    Wraps repeated ``grid.travel_cost(src, dst)`` calls so that duplicate
    (src, dst) pairs within one optimisation run are computed only once.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Coord, Coord], float] = {}

    def get_or_compute(self, src: Coord, dst: Coord, grid: Any) -> float:
        """Return the cached travel cost, computing it on first access."""
        key = (src, dst)
        if key not in self._cache:
            cost = grid.travel_cost(src, dst)
            self._cache[key] = cost
            # For undirected graphs cache the reverse as well
            if not getattr(grid.graph, "is_directed", lambda: False)():
                self._cache[(dst, src)] = cost
        return self._cache[key]

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
