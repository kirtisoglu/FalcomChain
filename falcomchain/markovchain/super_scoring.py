"""
Cut-scoring functions for the supergraph (level-2) cut phase.

The capacitated tree-cut phase at level 2 selects an admissible subtree
T_u with probability proportional to a candidate-awareness score

    ψ²(T_u) = ϕ²(T_u) · exp(-γ · η²(T_u))

where ϕ²(T_u) is the level-2 feasibility score (= the assigned capacity
when admissible) and η²(T_u) is the level-2 geometric penalty. Consistent
with the disaggregated hierarchical median objective (paper Section 5.4),
η² is the demand-weighted MEAN distance from the superdistrict's
demand-weighted 1-median (over base units) to the base units it serves —
the per-capita upper-level coordination cost, the level-2 analogue of the
base-level access penalty η¹.

This module provides the default implementation and a factory that binds
it to a chain state. Pass ``super_psi_fn=None`` to ``hierarchical_recom``
to fall back to the γ=0 case (ϕ²(T_u) = teams, uniform weighting over
admissible cuts).
"""

import math
from typing import Callable


def hub_coherence_psi_factory(
    state,
    *,
    gamma: float,
) -> Callable:
    """
    Build a level-2 cut-score function bound to ``state``.

    The returned callable has signature ``(subnodes, teams) -> float``
    suitable for ``CutParams.super_psi_fn``. ``subnodes`` is a frozenset
    of supergraph nodes (level-1 district IDs) in the candidate
    extraction; ``teams`` is the assigned capacity (= ϕ²(T_u)).

    Level-2 geometric penalty (demand-weighted 1-median, per-capita):

    .. math::

        \\eta^2(T_u) = \\frac{1}{D(T_u)}\\,
            \\min_{f \\in F^2 \\cap V^1[T_u]} \\;
            \\sum_{v \\in V^1[T_u]} d_v \\, \\mathrm{d}(f, v),
        \\qquad D(T_u) = \\sum_{v \\in V^1[T_u]} d_v.

    For each level-2 candidate ``f`` inside the subtree's base-level
    footprint, the inner sum is the total demand-weighted distance from
    ``f`` to every base unit. The outer ``min`` picks the candidate with
    the smallest such cost (the demand-weighted 1-median), and dividing by
    total demand yields the per-capita coordination cost. When no
    super-candidate is in the subtree, ``η²(T_u) = +∞`` and the score
    becomes 0 — the cut is filtered out by ``bipartition_tree``'s
    ``c.psi > 0`` check (soft constraint at level 2).

    :param state: The current :class:`~falcomchain.markovchain.ChainState`.
        Must have ``partition.parts`` (level-1 district -> base nodes) and
        ``assignment.travel_times``.
    :param gamma: Inverse-temperature parameter γ ≥ 0. ``gamma=0``
        reduces to ϕ²(T_u) (uniform weighting).
    :returns: A callable ``(subnodes, teams) -> float``.
    """
    partition = state.partition
    travel_times = state.assignment.travel_times
    graph_nodes = partition.graph.nodes

    def super_psi(subnodes, teams):
        if not subnodes:
            return 0.0

        # Base-level footprint: V^1[T_u] = ⋃_{v ∈ T_u} D_v^1
        base_nodes = set()
        for sg_node in subnodes:
            if sg_node in partition.parts:
                base_nodes |= partition.parts[sg_node]
        if not base_nodes:
            return 0.0

        # F^2 ∩ V^1[T_u]
        super_candidates = [
            v for v in base_nodes if graph_nodes[v].get("super_candidate", 0)
        ]
        if not super_candidates:
            return 0.0  # ψ² = 0 → cut excluded (soft level-2 constraint)

        if gamma == 0.0:
            # ψ² = ϕ²(T_u) when γ = 0; the soft existence check above
            # has already excluded subtrees with no super-candidate.
            return float(teams)

        # η²(T_u): demand-weighted 1-median over base units, per capita.
        total_demand = sum(graph_nodes[v]["demand"] for v in base_nodes)
        if total_demand <= 0:
            return 0.0
        best_cost = float("inf")
        for f in super_candidates:
            try:
                cost = sum(
                    graph_nodes[v]["demand"] * travel_times[(f, v)]
                    for v in base_nodes
                )
            except KeyError:
                continue
            if cost < best_cost:
                best_cost = cost
        if best_cost == float("inf"):
            return 0.0

        return float(teams) * math.exp(-gamma * best_cost / total_demand)

    return super_psi
