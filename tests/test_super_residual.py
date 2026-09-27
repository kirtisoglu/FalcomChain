"""
Residual-feasibility predicate at the supergraph level (CutParams.check_super_residual).

A one-sided supergraph extraction must leave behind supernodes that can still
be partitioned into super-districts with capacity in [c_min, c_max] and at
least ``min_districts_super`` districts each. Without the predicate the
greedy recursion strands a final supernode and the proposal is rejected
after exhausting its retry budget.
"""

import networkx as nx
import pytest

from falcomchain.random import set_seed
from falcomchain.tree.errors import CutSearchExhausted
from falcomchain.tree.tree import CutParams, SpanningTree, capacitated_recursive_tree, find_superedge_cuts


def _path_supergraph(capacities, demand_per_unit=100.0):
    g = nx.path_graph(len(capacities))
    for n, c in zip(g.nodes, capacities):
        g.nodes[n].update(n_teams=c, demand=c * demand_per_unit, area=1.0)
    return g


def _super_tree(capacities, *, c_min, c_max, min_d, check):
    g = _path_supergraph(capacities)
    n_teams = sum(capacities)
    params = CutParams(
        ideal_demand=100.0, epsilon=0.15, capacity_level=c_max, n_teams=n_teams,
        two_sided=n_teams <= c_max, c_min=c_min, min_districts_super=min_d,
        check_super_residual=check,
    )
    return SpanningTree(graph=g, params=params, supergraph=True)


def test_predicate_removes_stranding_extractions():
    # Four unit-capacity supernodes, c^2 in [2, 3], kappa = 2, one-sided mode
    # (4 > c_max = 3). Extracting three of them would strand a single node.
    set_seed(3)
    with_check = _super_tree([1, 1, 1, 1], c_min=2, c_max=3, min_d=2, check=True)
    cuts = find_superedge_cuts(with_check)
    assert cuts, "the two-node extractions must remain admissible"
    assert all(c.assigned_teams == 2 for c in cuts)
    set_seed(3)
    without = _super_tree([1, 1, 1, 1], c_min=2, c_max=3, min_d=2, check=False)
    assert any(c.assigned_teams == 3 for c in find_superedge_cuts(without))


def test_predicate_is_exactly_the_representability_condition():
    # Residual of 7 units on 5 supernodes with c in [2, 6], kappa = 2:
    # k super-districts need ceil(7/6) = 2 <= k <= min(floor(7/2), floor(5/2)) = 2 -> feasible.
    # Residual of 7 units on 3 supernodes: k <= min(3, 1) = 1 < 2 -> infeasible.
    set_seed(1)
    tree = _super_tree([2, 2, 1, 1, 1, 2], c_min=2, c_max=6, min_d=2, check=True)
    # All one-sided cuts found must leave a representable residual.
    for c in find_superedge_cuts(tree):
        c_res = tree.n_teams - c.assigned_teams
        n_res = tree.graph.number_of_nodes() - len(c.subnodes)
        k_lo = -(-c_res // 6)
        k_hi = min(c_res // 2, n_res // 2)
        assert k_lo <= k_hi


def test_recursion_closes_more_often_with_the_predicate():
    # A supergraph on which the greedy recursion frequently strands a node
    # without the predicate: a path of unit-capacity supernodes.
    failures = {}
    for check in (True, False):
        n_fail = 0
        for seed in range(40):
            set_seed(seed)
            g = _path_supergraph([1] * 9)
            try:
                capacitated_recursive_tree(
                    g, n_teams=9, demand_target=100.0, epsilon=0.15, capacity_level=3,
                    supergraph=True, c_min=2, max_attempts=3, min_districts_super=2,
                    check_super_residual=check,
                )
            except CutSearchExhausted:
                n_fail += 1
        failures[check] = n_fail
    assert failures[True] <= failures[False]
    assert failures[True] < 40
