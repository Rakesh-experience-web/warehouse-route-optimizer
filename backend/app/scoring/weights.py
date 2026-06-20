"""weights.py — Centralised heuristic weights and penalty coefficients.

Every numeric constant used in scoring, penalty, and heuristic logic lives
here.  Import :data:`DEFAULT_WEIGHTS` for sensible production defaults, or
construct a custom :class:`ScoringWeights` instance for tuning experiments.

The field values correspond directly to the ``OptimizationConfig`` parameters
of the same name so that callers can bridge between them via
:func:`from_config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.schemas import OptimizationConfig


@dataclass
class ScoringWeights:
    """Named constants for all heuristic scoring terms.

    Each field carries a docstring that explains the economic intuition behind
    the coefficient so that future tuners know what they are adjusting.
    """

    # ------------------------------------------------------------------
    # Distance and route cost
    # ------------------------------------------------------------------

    alpha_distance: float = 1.0
    """Weight applied to the Euclidean/centroid distance between an order and a
    batch medoid.  Increase to prefer spatially compact batches."""

    route_cost_reweight_factor: float = 1.0
    """Multiplier on the marginal route-distance delta in insertion scoring.
    Values above 1.0 make the optimizer more aggressive about avoiding
    batches that add long detours."""

    # ------------------------------------------------------------------
    # Urgency / due-time
    # ------------------------------------------------------------------

    beta_due_time: float = 0.4
    """Penalty for assigning an order to a batch whose projected completion
    time exceeds the order's due window.  Increase to prioritise on-time
    delivery over route efficiency."""

    urgency_weight: float = 0.35
    """Alias used in DHOBR scoring for the delay penalty term."""

    # ------------------------------------------------------------------
    # Weight / capacity
    # ------------------------------------------------------------------

    gamma_weight: float = 0.2
    """Penalty for increasing the weight pressure on a batch.  Increase to
    distribute heavy orders more evenly across pickers."""

    # ------------------------------------------------------------------
    # Category / zone similarity
    # ------------------------------------------------------------------

    delta_similarity: float = 1.0
    """Bonus for assigning orders with overlapping SKU categories to the same
    batch.  Higher values cluster by category more aggressively."""

    similarity_batch_boost: float = 0.5
    """Additional category-similarity bonus in insertion scoring."""

    advanced_category_boost_weight: float = 5.0
    """Reward applied when candidate order shares categories with the batch
    (used in seed-distance and advanced-insertion modes)."""

    # ------------------------------------------------------------------
    # Zone penalties
    # ------------------------------------------------------------------

    zone_mismatch_penalty_weight: float = 50.0
    """Penalty per extra zone introduced when adding an order to a batch that
    already has a dominant zone.  Large values enforce strict zone grouping."""

    same_zone_nearby_boost_weight: float = 10.0
    """Reward for adding an order that shares the batch's dominant zone *and*
    whose pick locations are close to existing picks."""

    # ------------------------------------------------------------------
    # Aisle / spread
    # ------------------------------------------------------------------

    advanced_aisle_penalty_weight: float = 1.0
    """Penalty per new aisle introduced by a candidate order.  Encourages
    batches that can be served by traversing fewer aisles."""

    advanced_spread_penalty_weight: float = 2.5
    """Penalty proportional to how far a candidate's picks are from the
    current batch centroid.  Keeps picks geographically tight."""

    advanced_route_weight: float = 1.0
    """Weight on raw route-distance delta in seed-distance batching."""

    # ------------------------------------------------------------------
    # Fragility and temperature
    # ------------------------------------------------------------------

    fragile_bulky_penalty: float = 0.4
    """Added to score when a fragile order is placed with a bulky order (or
    vice versa).  Increase if physical damage is a concern."""

    temperature_zone_mismatch_penalty: float = 0.2
    """Penalty for routing a temperature-sensitive order through an ambient
    zone.  Increase for cold-chain compliance."""

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    priority_score_weight: float = 0.05
    """Score reduction per unit of order priority level.  VIP orders get a
    small negative (beneficial) adjustment to their insertion score."""

    # ------------------------------------------------------------------
    # Overflow / exception
    # ------------------------------------------------------------------

    overflow_assignment_penalty: float = 0.15
    """Extra cost for assigning an order to an overflow batch.  Ensures
    standard batches are filled before overflow is used."""

    # ------------------------------------------------------------------
    # DHOBR item-level weights
    # ------------------------------------------------------------------

    dhobr_route_weight: float = 1.0
    """Weight on the route-increase delta inside DHOBR insertion scoring."""

    dhobr_similarity_weight: float = 0.45
    """Weight on the average pairwise distance between new items and existing
    batch items inside DHOBR scoring."""

    dhobr_picker_load_weight: float = 0.25
    """Weight on the normalised picker-load pressure inside DHOBR scoring."""

    dhobr_delay_weight: float = 0.35
    """Weight on the lateness penalty inside DHOBR scoring."""

    dhobr_new_batch_bias: float = 0.0
    """Constant bias added to the cost of creating a new DHOBR batch.
    Positive values make the algorithm prefer adding to existing batches."""

    # ------------------------------------------------------------------
    # Singleton merge
    # ------------------------------------------------------------------

    advanced_singleton_merge_max_delta: Optional[float] = None
    """Maximum route-distance increase allowed when merging a singleton batch
    into another.  None means merges are always accepted when feasible."""


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def from_config(config: OptimizationConfig) -> ScoringWeights:
    """Build a :class:`ScoringWeights` from an :class:`OptimizationConfig`.

    This bridges the API-facing configuration (which clients send) and the
    internal weight structure used by the scoring engine.
    """
    return ScoringWeights(
        alpha_distance=config.alpha_distance,
        route_cost_reweight_factor=config.route_cost_reweight_factor,
        beta_due_time=config.beta_due_time,
        urgency_weight=config.dhobr_delay_weight,
        gamma_weight=config.gamma_weight,
        delta_similarity=config.delta_similarity,
        similarity_batch_boost=config.similarity_batch_boost,
        advanced_category_boost_weight=config.advanced_category_boost_weight,
        zone_mismatch_penalty_weight=config.zone_mismatch_penalty_weight,
        same_zone_nearby_boost_weight=config.same_zone_nearby_boost_weight,
        advanced_aisle_penalty_weight=config.advanced_aisle_penalty_weight,
        advanced_spread_penalty_weight=config.advanced_spread_penalty_weight,
        advanced_route_weight=config.advanced_route_weight,
        fragile_bulky_penalty=config.fragile_bulky_penalty,
        temperature_zone_mismatch_penalty=config.temperature_zone_mismatch_penalty,
        priority_score_weight=config.priority_score_weight,
        overflow_assignment_penalty=config.overflow_assignment_penalty,
        dhobr_route_weight=config.dhobr_route_weight,
        dhobr_similarity_weight=config.dhobr_similarity_weight,
        dhobr_picker_load_weight=config.dhobr_picker_load_weight,
        dhobr_delay_weight=config.dhobr_delay_weight,
        dhobr_new_batch_bias=config.dhobr_new_batch_bias,
        advanced_singleton_merge_max_delta=config.advanced_singleton_merge_max_delta,
    )


#: Production defaults — mirrors OptimizationConfig field defaults.
DEFAULT_WEIGHTS = ScoringWeights()
