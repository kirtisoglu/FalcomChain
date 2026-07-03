from collections import namedtuple
from dataclasses import replace
from functools import partial
from typing import Callable, Optional, Tuple

from falcomchain.random import rng

from falcomchain.partition import Partition
from falcomchain.tree.tree import (
    Cut,
    Flip,
    ReselectException,
    bipartition_tree,
    capacitated_recursive_tree,
)

from .super_partitioners import resample_super_partition

class MetagraphError(Exception):
    """
    Raised when the partition we are trying to split is a low degree
    node in the metagraph.
    """

    pass


class ValueWarning(UserWarning):
    """
    Raised whe a particular value is technically valid, but may
    cause issues with the algorithm.
    """

    pass


def hierarchical_recom(
    state,
    epsilon_base: float,
    epsilon_super: float,
    demand_target: float,
    density: Optional[float] = None,
    super_partitioner: Callable = resample_super_partition,
    gamma_base: float = 0.0,
    gamma_super: float = 0.0,
    psi_fn: Optional[Callable] = None,
    super_psi_fn: Optional[Callable] = None,
    travel_times: Optional[dict] = None,
    c_min_base: int = 1,
    c_min_super: Optional[int] = None,
    c_max_super: Optional[int] = None,
    min_districts_super: int = 1,
    max_attempts_super: int = 1000,
    rule: str = "per_team",
    enforce_global_balance: bool = False,
):
    """
    Proposes a new ChainState via two-level hierarchical ReCom.

    The upper-level (supergraph) partitioning is delegated to
    ``super_partitioner``. The default :func:`resample_super_partition` samples
    a fresh level-2 partition each step. Pass :func:`fixed_super_partition`
    (or any custom callable with the same signature) for alternative behavior
    such as fixed superdistricts.

    Once the upper level returns the chosen superdistrict's level-1 IDs,
    those base nodes are re-partitioned via recursive partitioning to
    produce the new level-1 districts. Recording (if enabled) is driven by
    the ``Recorder`` attached to the chain via ``state._recorder``.

    :param state: The current ChainState.
    :param epsilon_base: Level-1 demand-balance tolerance ε¹ (paper). Used
        for the lower-level recursive partitioning of the chosen superdistrict.
    :param epsilon_super: Level-2 demand-balance tolerance ε² (paper). Used
        by the upper-level ``super_partitioner``.
    :param demand_target: Per-team workload d̄ (paper).
    :param density: Optional density parameter.
    :param super_partitioner: Callable with signature
        ``(state, epsilon_super, demand_target, density, recorder,
        super_psi_fn, gamma_super) ->
        (super_flip, merge, super_teams, super_demand, log_super_ratio)``.
        Defaults to :func:`resample_super_partition`.
    :param gamma_super: Inverse temperature γ² for the level-2 hub-coherence
        score (paper Eq. 22, 27). When ``> 0`` and ``super_psi_fn`` is
        ``None``, a hub-coherence scorer is built from ``state``. Default 0.
    :param super_psi_fn: Optional precomputed level-2 scorer
        ``(subnodes, teams) -> float``. Overrides the default factory built
        from ``gamma_super``.
    :param c_min_base: Minimum level-1 capacity per district (default 1).
    :param c_min_super: Minimum level-2 capacity (number of level-1
        districts per super-district). Defaults to ``2 · c_min_base`` —
        the standard choice that keeps super-districts non-trivial and
        aligns the supergraph's leaf demand range with the actual
        super-node weights (≈ ``c¹·w``).
    :param c_max_super: Maximum level-2 capacity (paper §6.4 default: 3).
        Required.
    :returns: The proposed ChainState.
    """
    if c_min_super is None:
        c_min_super = 2 * c_min_base
    if c_min_super < 2 * c_min_base:
        raise ValueError(
            f"c_min_super (={c_min_super}) must be ≥ 2·c_min_base "
            f"(={2 * c_min_base}). Tighter values produce trivial "
            f"identity-grouping super-partitions where no merge is admissible."
        )
    # Default c_max_super to 2·c¹_max so super-leaves can fit two
    # max-capacity L1 districts. Setting it equal to c¹_max (paper's
    # narrow choice of 3) leaves room only for single-super-node leaves
    # and the recursion typically can't find admissible cuts.
    if c_max_super is None:
        c_max_super = max(3, 2 * state.partition.capacity_level)
    partition = state.partition
    rec = getattr(state, "_recorder", None)

    # Signal start of a new step to the recorder
    if rec is not None:
        rec.begin_step()

    method = partial(
        capacitated_recursive_tree,
        capacity_level=partition.capacity_level,
        density=density,
        recorder=rec,
        c_min=c_min_base,
        gamma=gamma_base,
        travel_times=travel_times,
        psi_fn=psi_fn,
        rule=rule,
        enforce_global_balance=enforce_global_balance,
    )

    # ---- UPPER LEVEL: delegated to the super_partitioner callable ----
    if rec is not None:
        rec.begin_level("supergraph", partition=partition)

    super_flip, merge, super_teams, super_demand, log_super_ratio = (
        super_partitioner(
            state,
            epsilon_super=epsilon_super,
            demand_target=demand_target,
            c_min_super=c_min_super,
            c_max_super=c_max_super,
            density=density,
            recorder=rec,
            super_psi_fn=super_psi_fn,
            gamma_super=gamma_super,
            max_attempts=max_attempts_super,
            min_districts_super=min_districts_super,
        )
    )

    if rec is not None:
        super_centers = {}
        if state.super_facility:
            super_centers = dict(state.super_facility.centers)
        rec.end_level(centers=super_centers)

    # Carry the full level-2 assignment forward so the new partition
    # can persist it (used by SuperFacilityAssignment, ensemble analysis,
    # and any future level-2 consumer).
    superflip = replace(super_flip, merged_ids=frozenset(merge))

    merged_base_nodes = set.union(*(set(partition.parts[part]) for part in merge))
    subgraph = partition.graph.graph.subgraph(merged_base_nodes)

    if rec is not None:
        rec.record_select(
            supergraph=partition.supergraph,
            selected_superdistricts=merge,
            merged_base_nodes=merged_base_nodes,
            partition=partition,
        )

    # ---- LOWER LEVEL: Re-partition base subgraph H ----
    new_demand_target = super_demand / super_teams if super_teams else demand_target

    max_id = max(district for district in partition.parts)
    sub_assignments = {
        node: partition.assignment.mapping[node] for node in subgraph.nodes
    }

    if rec is not None:
        rec.begin_level("base")

    # Initial debt is 0 by construction at the lower level: new_demand_target
    # is recomputed locally as super_demand / super_teams (Remark A.5 in the
    # paper). No initial-debt argument is passed.
    flip = method(
        graph=subgraph,
        n_teams=int(super_teams),
        merged_ids=set(merge.copy()),
        assignments=sub_assignments,
        max_id=max_id,
        demand_target=new_demand_target,
        epsilon=epsilon_base,
        iteration=partition.step,
    )
    flip = flip.add_merged_ids(merge)

    # Pure Boltzmann acceptance: ignore forward proposal density since we
    # don't compute the reverse term (see GerryChain, Cannon et al. 2022).
    # Setting log_proposal_ratio=0 gives alpha = exp(-beta * delta_E).
    # The forward ratio is still stored for diagnostics/future reversible variant.
    forward_log_proposal = log_super_ratio + flip.log_proposal_ratio

    proposed_partition = partition.perform_flip(flipp=flip, superflipp=superflip)

    # FalCom is a sampler by default. Energy is an objective function used
    # only by the optimizer (boltzmann acceptance) and is therefore opt-in:
    # when state.energy_fn is None we leave proposed_state.energy at the
    # placeholder, never invoking compute_energy. Users who want Boltzmann
    # optimization pass energy_fn=compute_energy (or a custom callable) to
    # ChainState.initial.
    proposed_state = state.next(
        partition=proposed_partition,
        energy=0.0,
        log_proposal_ratio=0.0,
        feasible=True,
    )

    if rec is not None:
        # Level-1 facility centers
        level1_centers = {}
        if proposed_state.facility:
            level1_centers = dict(proposed_state.facility.centers)
        rec.end_level(centers=level1_centers)

    # ---- ACCEPT/REJECT ----
    if rec is not None:
        rec.record_accept_reject(
            proposed_state=proposed_state,
            current_state=state,
            accepted=True,  # tentative — MarkovChain sets final value
        )
    return proposed_state


def recom(  # Note: recomb is called for each state of the chain. Parameters must be static for the states. (should we cache some of them in Partition?)
    partition: Partition,
    demand_target: int,
    column_names: tuple[str],
    epsilon: float,
    density: float = None,
) -> Partition:
    """
    ReCom (short for ReCombination) is a Markov Chain Monte Carlo (MCMC) algorithm
    used for redistricting. At each step of the algorithm, a pair of adjacent districts
    is selected at random and merged into a single district. The region is then split
    into two new districts by generating a spanning tree using the Kruskal/Karger
    algorithm and cutting an edge at random. The edge is checked to ensure that it
    separates the region into two new districts that are demand balanced, and,
    if not, a new edge is selected at random and the process is repeated.

    :param partition: The initial partition.
    :type partition: Partition
    :param demand_col: The name of the demand column.
    :type demand_col: str
    :param demand_target: The target demand for each district.
    :type demand_target: Union[int,float]
    :param epsilon: The epsilon value for demand deviation as a percentage of the
        target demand.
    :type epsilon: float
    :param node_repeats: The number of times to repeat the bipartitioning step. Default is 1.
    :type node_repeats: int, optional

    :returns: The new partition resulting from the ReCom algorithm.
    :rtype: Partition
    """
    bad_district_pairs = set()
    n_parts = len(partition)
    tot_pairs = (
        n_parts * (n_parts - 1) / 2
    )  # n choose 2  (isn't it too big? no adjacency between any two districts. it should be # of super cut edges)
    ids = set(partition.parts.keys())

    while len(bad_district_pairs) < tot_pairs:
        try:
            while True:
                edge = rng.choice(tuple(partition["cut_edges"]))
                # Need to sort the tuple so that the order is consistent in the bad_district_pairs set
                part_one, part_two = (
                    partition.assignment.mapping[edge[0]],
                    partition.assignment.mapping[edge[1]],
                )
                parts_to_merge = [part_one, part_two]
                parts_to_merge.sort()

                if tuple(parts_to_merge) not in bad_district_pairs:
                    break

            n_teams = partition.teams[part_one] + partition.teams[part_two]
            subgraph = partition.graph.subgraph(
                partition.parts[part_one] | partition.parts[part_two]
            )

            flips, new_teams = capacitated_recursive_tree(
                graph=subgraph.graph,
                column_names=column_names,
                n_teams=n_teams,
                demand_target=demand_target,
                epsilon=epsilon,
                capacity_level=partition.capacity_level,
                density=density,
                assignments=partition.assignment,
                merged_parts=parts_to_merge,
                ids=ids,
            )
            break

        except Exception as e:
            if isinstance(
                e, ReselectException
            ):  # if there is no balanced cut after max_attempt in bipartition_tree, then the pair is a bad district pair.
                bad_district_pairs.add(tuple(parts_to_merge))
                continue
            else:
                raise

    if len(bad_district_pairs) == tot_pairs:
        raise MetagraphError(
            f"Bipartitioning failed for all {tot_pairs} district pairs."
            f"Consider rerunning the chain with a different random seed."
        )

    return partition.flip(flips, new_teams)


def propose_chunk_flip(partition: Partition) -> Partition:
    """
    Chooses a random boundary node and proposes to flip it and all of its neighbors

    :param partition: The current partition to propose a flip from.
    :type partition: Partition

    :returns: A possible next `~falcomchain.partition.Partition`
    :rtype: Partition
    """
    flips = dict()

    edge = rng.choice(tuple(partition["cut_edges"]))
    index = rng.choice((0, 1))

    flipped_node = edge[index]

    valid_flips = [
        nbr
        for nbr in partition.graph.neighbors(flipped_node)
        if partition.assignment.mapping[nbr]
        != partition.assignment.mapping[flipped_node]
    ]

    for flipped_neighbor in valid_flips:
        flips.update({flipped_neighbor: partition.assignment.mapping[flipped_node]})

    return partition.flip(flips)


def propose_random_flip(partition: Partition) -> Partition:
    """
    Proposes a random boundary flip from the partition.

    :param partition: The current partition to propose a flip from.
    :type partition: Partition

    :returns: A possible next `~falcomchain.partition.Partition`
    :rtype: Partition
    """
    if len(partition["cut_edges"]) == 0:
        return partition
    edge = rng.choice(tuple(partition["cut_edges"]))
    index = rng.choice((0, 1))
    flipped_node, other_node = edge[index], edge[1 - index]
    flip = {flipped_node: partition.assignment.mapping[other_node]}
    return partition.flip(flip)
