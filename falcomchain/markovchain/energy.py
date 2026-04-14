from falcomchain.markovchain.state import ChainState


def compute_energy(state: ChainState) -> float:
    """
    Compute the accessibility energy of a partition state.

    E(s) = ΣᴅΣᵥ dᵥ · dist(v, f(D))

    where:
      - D ranges over all districts
      - v ranges over all nodes in district D
      - dᵥ is the demand (population) of node v
      - f(D) is the chosen facility center of district D
      - dist(v, f(D)) is the travel time from center f(D) to node v

    Travel times are read from ``Assignment.travel_times``, a class-level
    dict keyed by ``(facility_node, node)``.

    :param state: The current chain state.
    :type state: ChainState
    :returns: Total weighted travel time across all districts.
    :rtype: float
    """
    assignment = state.assignment
    travel_times = assignment.travel_times
    graph = state.graph
    centers = state.facility.centers

    total = 0.0
    for district, nodes in assignment.parts.items():
        center = centers.get(district)
        if center is None:
            continue
        for node in nodes:
            demand = graph.nodes[node]["demand"]
            total += demand * travel_times[(center, node)]

    return total


def compute_energy_delta(proposed: ChainState, current: ChainState) -> float:
    """
    Compute E(proposed) - E(current) efficiently by re-evaluating only
    the districts that changed (those in ``proposed.partition.flow.node_flows``).

    :param proposed: The proposed chain state.
    :param current:  The current chain state.
    :returns: Delta energy E(s') - E(s).
    :rtype: float
    """
    assignment = proposed.assignment
    travel_times = assignment.travel_times
    graph = proposed.graph

    flow = proposed.partition.flow
    changed_parts = set(flow.node_flows.keys()) if flow.node_flows else set()
    # Also include parts that gained/lost district IDs
    changed_parts |= flow.part_flows["in"] | flow.part_flows["out"]

    def district_energy(parts_dict, centers_dict, district):
        center = centers_dict.get(district)
        if center is None:
            return 0.0
        return sum(
            graph.nodes[node]["demand"] * travel_times[(center, node)]
            for node in parts_dict.get(district, frozenset())
        )

    delta = 0.0
    proposed_centers = proposed.facility.centers
    current_centers = current.facility.centers
    current_parts = current.assignment.parts

    for district in changed_parts:
        new_e = district_energy(assignment.parts, proposed_centers, district)
        old_e = district_energy(current_parts, current_centers, district)
        delta += new_e - old_e

    return delta
