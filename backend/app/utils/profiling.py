"""utils/profiling.py — Timing decorators and profiling context managers."""
from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def timed(label: str | None = None) -> Callable[[F], F]:
    """Decorator that logs execution time of the wrapped function.

    Args:
        label: Custom label for the log message. Defaults to the function name.

    Example::

        @timed("batching")
        def run_batching(...): ...
    """
    def decorator(fn: F) -> F:
        name = label or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug("timed label=%s elapsed_ms=%.2f", name, elapsed_ms)
                return result
            except Exception:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug("timed label=%s elapsed_ms=%.2f status=error", name, elapsed_ms)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def profiling_block(label: str) -> Generator[None, None, None]:
    """Context manager that logs execution time of an arbitrary code block.

    Example::

        with profiling_block("route_estimation"):
            cost = estimate_route(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("profiling_block label=%s elapsed_ms=%.2f", label, elapsed_ms)
