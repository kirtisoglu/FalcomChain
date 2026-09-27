"""
The counting predicate (paper: level-1 admissibility predicate) lets the
recursive partitioner run on sparse, real-world candidate sets -- far fewer
candidates than Assumption 6.1 asks for -- and the chain reports why
proposals were rejected.
"""
import math
from functools import partial

import pytest

from falcomchain import MarkovChain, Partition, always_accept, hierarchical_recom
from falcomchain.candidates.feasibility import check_facility_density
from falcomchain.graph.grid import Grid
from falcomchain.markovchain.chain import classify_rejection
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed
from falcomchain.tree.errors import CutSearchExhausted, PopulationBalanceError


@pytest.fixture
def sparse_grid():
    """20x20 grid, 12 candidates for ~10 districts: Assumption 6.1 fails badly."""
    set_seed(7)
    graph = Grid(dimensions=(20, 20), num_candidates=12, density="uniform").graph
    Assignment.travel_times = {
        (a, b): float(abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
                      + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"]))
        for a in graph.nodes for b in graph.nodes
    }
    return graph


def _targets(graph, k=10):
    total = sum(d["demand"] for _, d in graph.nodes(data=True))
    w = total / k
    return w, total / math.ceil(total / w)


def test_assumption_fails_on_the_sparse_grid(sparse_grid):
    w, _ = _targets(sparse_grid)
    assert not check_facility_density(sparse_grid, demand_target=w, epsilon=0.15).passes


def test_initial_partition_needs_the_counting_predicate(sparse_grid):
    w, seed_target = _targets(sparse_grid)
    set_seed(1)
    with pytest.raises(CutSearchExhausted) as info:
        Partition.from_random_assignment(
            graph=sparse_grid, epsilon=0.15, demand_target=seed_target,
            assignment_class=None, capacity_level=2,
            count_candidates=False, max_attempts=100,
        )
    assert info.value.level == "base"
    assert isinstance(info.value, RuntimeError)   # the chain treats it as a rejection

    set_seed(1)
    partition = Partition.from_random_assignment(
        graph=sparse_grid, epsilon=0.15, demand_target=seed_target,
        assignment_class=None, capacity_level=2,   # counting is the default
    )
    assert all(partition.assignment.candidates[p] for p in partition.parts)


def test_chain_runs_on_sparse_candidates_and_reports_rejections(sparse_grid):
    w, seed_target = _targets(sparse_grid)
    set_seed(1)
    partition = Partition.from_random_assignment(
        graph=sparse_grid, epsilon=0.15, demand_target=seed_target,
        assignment_class=None, capacity_level=2,
    )
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    chain = MarkovChain(
        proposal=partial(hierarchical_recom, epsilon_base=0.15, epsilon_super=0.15,
                         demand_target=w, max_attempts_base=200),
        constraints=lambda p: True, accept=always_accept,
        initial_state=state, total_steps=40,
    )
    n_accepted = 0
    for i, st in enumerate(chain):
        if i > 0 and st is not prev:
            n_accepted += 1
        prev = st
        # every district of every visited state contains a candidate
        assert all(st.partition.assignment.candidates[p] for p in st.partition.parts)
    report = chain.rejection_report()
    assert report["steps"] == 39
    assert report["accepted"] == n_accepted
    assert report["accepted"] + report["rejected"] == 39
    assert n_accepted > 0
    assert set(report["causes"]) <= {
        "cut_search_exhausted:base", "cut_search_exhausted:super",
        "balance_violation", "runtime_error",
    }


def test_classify_rejection_labels():
    assert classify_rejection(CutSearchExhausted("base", 5)) == "cut_search_exhausted:base"
    assert classify_rejection(CutSearchExhausted("super", 5)) == "cut_search_exhausted:super"
    assert classify_rejection(PopulationBalanceError("x")) == "balance_violation"
    assert classify_rejection(RuntimeError("x")) == "runtime_error"
    assert "Supergraph = True" in str(CutSearchExhausted("super", 5))
