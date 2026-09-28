"""
Every sampled state must satisfy the kappa constraint: each super-district
bundles at least ``min_districts_super`` districts (paper constraint
(cons:minL1perL2)). The supergraph cut enforces it, and the base-level re-cut
may not undo it.
"""
from functools import partial

import networkx as nx

from falcomchain import MarkovChain, always_accept, hierarchical_recom
from falcomchain.graph import Graph
from falcomchain.markovchain.chain import classify_rejection
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed
from falcomchain.tree.errors import SuperDistrictTooSmall
from falcomchain.tree.tree import Flip


def _grid_3x4():
    g = nx.grid_2d_graph(3, 4)
    g = nx.relabel_nodes(g, {(r, c): r * 4 + c for r in range(3) for c in range(4)})
    cand = {0, 3, 6, 9}
    for n in g.nodes:
        r, c = divmod(n, 4)
        g.nodes[n].update(demand=100.0, area=1.0, C_X=float(c), C_Y=float(r),
                          candidate=1 if n in cand else 0, boundary_node=False, boundary_perim=0)
    return g


def test_lower_level_recut_cannot_leave_a_super_district_below_kappa():
    g = _grid_3x4()
    Assignment.travel_times = {(u, v): abs(g.nodes[u]["C_X"] - g.nodes[v]["C_X"])
                               + abs(g.nodes[u]["C_Y"] - g.nodes[v]["C_Y"])
                               for u in g.nodes for v in g.nodes}
    graph = Graph.from_networkx(g)
    # four unit-capacity districts of three nodes each (w = 300, eps = 0.15)
    flips = {n: 1 + (n % 4 // 2) + 2 * (n // 8) if False else None for n in g.nodes}
    districts = [[0, 1, 2], [3, 7, 11], [4, 8, 9], [5, 6, 10]]
    flips = {n: i + 1 for i, d in enumerate(districts) for n in d}
    teams = {i + 1: 1 for i in range(4)}
    flip = Flip(flips=flips, team_flips=teams, new_ids=frozenset(teams), merged_ids=frozenset())
    partition = Partition(capacity_level=2, assignment=flips, graph=graph, flip=flip)
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    set_seed(11)
    chain = MarkovChain(
        proposal=partial(hierarchical_recom, epsilon_base=0.15, epsilon_super=0.15,
                         demand_target=300.0, c_min_super=2, c_max_super=4,
                         min_districts_super=2, max_attempts_base=200, max_attempts_super=200),
        constraints=[], accept=always_accept, initial_state=state, total_steps=400,
    )
    for i, st in enumerate(chain):
        if i == 0:
            continue    # the identity level-2 start is replaced by the first accepted step
        for ids in st.partition.super_parts.values():
            assert len(ids) >= 2, f"step {i}: super-district with {len(ids)} district(s)"
    causes = chain.rejection_report()["causes"]
    assert causes.get("super_district_below_kappa", 0) > 0   # the case does occur and is rejected


def test_classification_label():
    assert classify_rejection(SuperDistrictTooSmall("x")) == "super_district_below_kappa"
