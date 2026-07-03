"""
Pluggable upper-level (supergraph) partitioners for ``hierarchical_recom``.

A super-partitioner produces, for each chain step:

- ``super_flip``: optional :class:`~falcomchain.tree.tree.Flip` carrying the
  full level-2 assignment over the parent's level-1 district IDs (or
  ``None`` if no full assignment is available, e.g. the bipartition fallback).
- ``merge``: ``frozenset`` of level-1 district IDs in the chosen
  superdistrict — the IDs that get merged and re-partitioned at the base level.
- ``super_teams``: total number of teams in the chosen superdistrict.
- ``super_demand``: total demand in the chosen superdistrict.
- ``log_super_ratio``: log proposal-ratio contribution from the upper level.

Two builtins:

- :func:`resample_super_partition` (default): every step samples a fresh
  level-2 partition via recursive partitioning of the supergraph; bipartition
  fallback if recursive fails.
- :func:`fixed_super_partition`: the level-2 partition is fixed; each step
  picks one superdistrict uniformly at random and refines its base-level
  districts. Use when superdistricts are externally given (e.g. health zones).

Custom partitioners with the same signature can be plugged in directly.
"""

from typing import Callable, Optional, Tuple

from falcomchain.random import rng
from falcomchain.tree.tree import Flip, capacitated_recursive_tree

from .super_scoring import hub_coherence_psi_factory


def resample_super_partition(
    state,
    epsilon_super: float,
    demand_target: float,
    c_min_super: int,
    c_max_super: int,
    density: Optional[float] = None,
    recorder=None,
    super_psi_fn: Optional[Callable] = None,
    gamma_super: float = 0.0,
    max_attempts: int = 1000,
    min_districts_super: int = 1,
) -> Tuple[Flip, frozenset, int, float, float]:
    """
    Default super-partitioner: resample the level-2 partition each step.

    Recursive-partitioning of the supergraph (Algorithm 2, line 2 in the
    paper) using :func:`~falcomchain.tree.tree.capacitated_recursive_tree`
    with the level-2 capacity bounds ``[c_min_super, c_max_super]``. One
    super-district is then selected uniformly at random as the merge set.

    Both level-2 capacity bounds are required: leaf super-districts must
    have demand in ``[(1−ε²)·c²·w, (1+ε²)·c²·w]`` for some
    ``c² ∈ [c_min_super, c_max_super]``, and these bounds must align with
    the actual super-node weights (which are the parent's level-1 district
    demands, ≈ ``c¹·w`` per node). The standard choice is
    ``c²_min = 2·c¹_min`` so that a super-district holds at least two
    level-1 districts; setting ``c²_min = 1`` produces the trivial
    "every super-district is a single level-1 district" identity grouping
    and the recursion typically fails to find admissible cuts.

    :param epsilon_super: Level-2 demand-balance tolerance ε² (paper).
    :param c_min_super: Minimum level-2 capacity (number of level-1
        districts per super-district). Must be ≥ 1; recommended
        ``2 · c_min_base``.
    :param c_max_super: Maximum level-2 capacity. Paper §6.4 default: 3.
    :param super_psi_fn: Optional precomputed ψ² scorer
        ``(subnodes, teams) -> float``. When ``None`` and ``gamma_super>0``,
        a hub-coherence scorer (paper Eq. 27) is built from ``state``. When
        ``gamma_super=0``, ψ²(T_u) = ϕ²(T_u) = teams (paper γ=0 case).
    :param gamma_super: Inverse temperature γ² ≥ 0 for the hub-coherence
        scorer. Ignored when ``super_psi_fn`` is provided.

    :returns: ``(super_flip, merge, super_teams, super_demand, log_super_ratio)``
    """
    partition = state.partition
    total_teams = sum(partition.teams.values())

    if super_psi_fn is None and gamma_super > 0.0:
        super_psi_fn = hub_coherence_psi_factory(state, gamma=gamma_super)

    super_flip = capacitated_recursive_tree(
        graph=partition.supergraph.copy(),
        n_teams=total_teams,
        demand_target=demand_target,
        epsilon=epsilon_super,
        capacity_level=c_max_super,
        c_min=c_min_super,
        density=density,
        supergraph=True,
        iteration=partition.step,
        recorder=recorder,
        super_psi_fn=super_psi_fn,
        max_attempts=max_attempts,
        min_districts_super=min_districts_super,
    )
    log_super_ratio = super_flip.log_proposal_ratio

    # Invert: super_id -> set of supergraph nodes (= level-1 district IDs)
    super_parts = {}
    for sg_node, super_id in super_flip.flips.items():
        super_parts.setdefault(super_id, set()).add(sg_node)

    chosen_super_id = rng.choice(list(super_parts.keys()))
    merge = frozenset(super_parts[chosen_super_id])
    super_teams = super_flip.team_flips[chosen_super_id]
    super_demand = sum(
        partition.supergraph.nodes[n].get("demand", 0) for n in merge
    )

    return super_flip, merge, super_teams, super_demand, log_super_ratio


def fixed_super_partition(
    state,
    epsilon_super: float,
    demand_target: float,
    c_min_super: int = 1,
    c_max_super: int = 1,
    density: Optional[float] = None,
    recorder=None,
    super_psi_fn: Optional[Callable] = None,
    gamma_super: float = 0.0,
    max_attempts: int = 1000,
    min_districts_super: int = 1,
) -> Tuple[Flip, frozenset, int, float, float]:
    """
    Fixed-superdistrict partitioner: never resamples the level-2 partition.

    Picks one superdistrict uniformly at random (from
    ``state.partition.super_parts``) and returns the level-1 districts in it
    as the merge set. The level-2 assignment passed forward is the parent's,
    unchanged.

    Use this when superdistricts are externally given (e.g. health zones,
    administrative regions) and should not be resampled. The chain then
    explores the level-1 partition within each fixed superdistrict.

    ``epsilon_super``, ``c_min_super``, ``c_max_super``, ``super_psi_fn``,
    ``gamma_super``, and ``min_districts_super`` are accepted for API parity
    with :func:`resample_super_partition` but ignored — no supergraph cuts
    are made when the level-2 partition is fixed.

    :returns: ``(super_flip, merge, super_teams, super_demand, log_super_ratio=0.0)``
    """
    del epsilon_super, c_min_super, c_max_super
    del super_psi_fn, gamma_super, max_attempts, min_districts_super  # unused — fixed mode
    partition = state.partition
    super_parts = partition.super_parts
    if not super_parts:
        raise ValueError(
            "fixed_super_partition called on a partition with empty "
            "super_parts. Did you forget to set super_assignment?"
        )

    chosen_super_id = rng.choice(list(super_parts.keys()))
    merge = frozenset(super_parts[chosen_super_id])

    super_teams = sum(partition.teams.get(d, 0) for d in merge)
    super_demand = sum(partition.part_demand(d) for d in merge)

    # Carry the parent's level-2 assignment forward unchanged.
    # Build the team_flips dict the same way Partition derives super_teams.
    team_flips = {}
    for l1_id, super_id in partition.super_assignment.items():
        team_flips[super_id] = team_flips.get(super_id, 0) + partition.teams.get(l1_id, 0)

    super_flip = Flip(
        flips=dict(partition.super_assignment),
        team_flips=team_flips,
        new_ids=frozenset(),
        merged_ids=frozenset(),
        log_proposal_ratio=0.0,
    )

    return super_flip, merge, super_teams, super_demand, 0.0
