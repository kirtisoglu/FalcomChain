from collections import namedtuple
from functools import partial
from typing import Optional, Tuple

from falcomchain.random import rng

from falcomchain.partition import Partition
from falcomchain.tree.tree import (
    Cut,
    Flip,
    ReselectException,
    bipartition_tree,
    capacitated_recursive_tree,
)

# imported lazily to avoid circular imports at module load time
def _get_chain_state_cls():
    from falcomchain.markovchain.state import ChainState
    from falcomchain.markovchain.energy import compute_energy
    return ChainState, compute_energy


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
    epsilon: float,
    demand_target: float,
    density: Optional[float] = None,
):
    """
    Proposes a new ChainState via two-level hierarchical ReCom.

    At the upper level, the supergraph is recursively partitioned into superdistricts.
    One superdistrict is chosen uniformly at random and its merged region is re-split
    by recursive partitioning at the base level. Recording (if enabled) is driven by
    the ``Recorder`` attached to the chain via ``state._recorder``.

    :param state: The current ChainState.
    :param epsilon: Maximum relative demand deviation allowed.
    :param demand_target: Target demand per team (d_bar).
    :param density: Optional density parameter.
    :returns: The proposed ChainState.
    """
    ChainState, compute_energy = _get_chain_state_cls()
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
    )

    # ---- UPPER LEVEL: Recursive partitioning of supergraph G² ----
    # Paper Algorithm 2, line 2: partition G^2 into superdistricts via
    # RecursivePartitioning. Falls back to single bipartition on small supergraphs
    # where the recursive approach can fail (e.g., teams don't split cleanly).
    total_teams = sum(partition.teams.values())

    if rec is not None:
        rec.begin_level("supergraph", partition=partition)

    try:
        super_flip = capacitated_recursive_tree(
            graph=partition.supergraph.copy(),
            n_teams=total_teams,
            demand_target=demand_target,
            epsilon=epsilon,
            capacity_level=partition.capacity_level,
            density=density,
            supergraph=True,
            iteration=partition.step,
            recorder=rec,
        )
        log_super_ratio = super_flip.log_proposal_ratio

        # Invert super_flip: superdistrict_id -> set of supergraph nodes
        super_parts = {}
        for sg_node, super_id in super_flip.flips.items():
            super_parts.setdefault(super_id, set()).add(sg_node)

        # Pick one superdistrict uniformly at random (Algorithm 1 line 3)
        chosen_super_id = rng.choice(list(super_parts.keys()))
        merge = frozenset(super_parts[chosen_super_id])
        super_teams = super_flip.team_flips[chosen_super_id]
        super_demand = sum(
            partition.supergraph.nodes[n].get("demand", 0) for n in merge
        )
    except RuntimeError:
        # Fallback for small supergraphs: single bipartition extraction
        acut_object, log_super_ratio = bipartition_tree(
            graph=partition.supergraph.copy(),
            demand_target=demand_target,
            capacity_level=partition.capacity_level,
            n_teams=total_teams,
            epsilon=epsilon,
            two_sided=False,
            supergraph=True,
            density=False,
            max_attempts=100,
            iteration=partition.step,
            recorder=rec,
        )
        merge = frozenset(acut_object.subnodes)
        super_teams = acut_object.assigned_teams
        super_demand = acut_object.demand

    if rec is not None:
        super_centers = {}
        if state.super_facility:
            super_centers = dict(state.super_facility.centers)
        rec.end_level(centers=super_centers)

    superflip = Flip(merged_ids=merge)

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

    flip = method(
        graph=subgraph,
        n_teams=int(super_teams),
        merged_ids=set(merge.copy()),
        assignments=sub_assignments,
        max_id=max_id,
        demand_target=new_demand_target,
        epsilon=epsilon,
        debt=(super_demand - super_teams * demand_target),
        iteration=partition.step,
    )
    flip = flip.add_merged_ids(merge)

    # Pure Boltzmann acceptance: ignore forward proposal density since we
    # don't compute the reverse term (see GerryChain, Cannon et al. 2022).
    # Setting log_proposal_ratio=0 gives alpha = exp(-beta * delta_E).
    # The forward ratio is still stored for diagnostics/future reversible variant.
    forward_log_proposal = log_super_ratio + flip.log_proposal_ratio

    proposed_partition = partition.perform_flip(flipp=flip, superflipp=superflip)

    # state.next() will use state.energy_fn if set, otherwise the energy arg below.
    # We default to compute_energy when no custom energy_fn is configured.
    if state.energy_fn is None:
        # Need to first build the state to compute default energy via FacilityAssignment
        proposed_state = state.next(
            partition=proposed_partition,
            energy=0.0,
            log_proposal_ratio=0.0,
            feasible=True,
        )
        proposed_state.energy = compute_energy(proposed_state)
    else:
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
