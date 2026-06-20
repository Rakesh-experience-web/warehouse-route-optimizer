"""_types.py — Shared dataclasses for the batching subpackage.

Kept in a private module so that all other batching modules can import these
lightweight types without creating circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

Coord = Tuple[int, int]


@dataclass
class BatchSummary:
    """Aggregated statistics for a tentative batch of orders.

    Computed by :func:`~app.optimizer.batching._summary.summarize_batch` and
    consumed by constraint validators and scoring functions.
    """

    order_count: int
    """Number of orders in the batch."""

    total_units: int
    """Sum of item quantities across all orders."""

    total_weight: float
    """Total pick weight (kg)."""

    total_volume: float
    """Total pick volume (m³ or warehouse-defined unit)."""

    target_count: int
    """Number of distinct shelf/pick-face positions to visit."""

    route_distance: float
    """Estimated travel distance for the batch route."""

    duration_seconds: float
    """Estimated pick duration in seconds."""

    zones: Set[str]
    """Warehouse zones visited by this batch."""

    fragile: bool
    """True when at least one order item is marked fragile."""

    bulky: bool
    """True when at least one order item is marked bulky."""


@dataclass
class BatchAssignment:
    """Output of a batching algorithm.

    Keeps the same public interface as the legacy class in batching.py for full
    backward compatibility.
    """

    batches: List[List["Order"]]  # type: ignore[name-defined]  # forward ref
    labels: Dict[str, int]
    batch_names: List[str] = field(default_factory=list)
    batch_types: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    exception_order_ids: List[str] = field(default_factory=list)
    batch_items: List[List["BatchItem"]] = field(default_factory=list)
    picker_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchItem:
    """A single line item resolved to a warehouse coordinate.

    Used by the DHOBR batching algorithm which works at item granularity
    rather than order granularity.
    """

    order_id: str
    sku: str
    qty: int
    coord: Coord
    category: str
    zone: str
    due_time_minutes: int
    created_at_epoch: int


@dataclass
class SmartBatch:
    """Live state maintained by the DHOBR item-level batcher."""

    batch_id: str
    items: List[BatchItem]
    route: List[Coord]
    picker_id: str
    current_load: int = 0
