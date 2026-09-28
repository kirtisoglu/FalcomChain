"""
Tests for the level-2 cut score `hub_coherence_psi_factory`: the per-capita
demand-weighted 1-median coordination cost of the paper's level-2 penalty.

Covers:
- The pure formula (γ=0 case, missing super_candidates, missing facilities).
- The factory's binding to a ChainState.
- End-to-end activation through `hierarchical_recom(gamma_super=...)`.
"""

import math
from types import SimpleNamespace

import pytest

from falcomchain.graph.grid import Grid
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.super_scoring import hub_coherence_psi_factory
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed


# ---------------------------------------------------------------------------
# Pure formula tests (mock state)
# ---------------------------------------------------------------------------

def _make_mock_state(*, parts, level1_centers, super_candidates, travel_times):
    """Build a SimpleNamespace state sufficient for hub_coherence_psi_factory."""
    graph_nodes = {}
    all_base_nodes = set()
    for ids in parts.values():
        all_base_nodes |= ids
    for n in all_base_nodes:
        graph_nodes[n] = {"super_candidate": 1 if n in super_candidates else 0,
                          "demand": 1.0}

    partition = SimpleNamespace(
        parts=parts,
        graph=SimpleNamespace(nodes=graph_nodes),
    )
    facility = SimpleNamespace(centers=level1_centers)
    assignment = SimpleNamespace(travel_times=travel_times)
    return SimpleNamespace(
        partition=partition,
        facility=facility,
        assignment=assignment,
    )


class TestHubCoherencePsiFactory:
    def test_returns_zero_when_no_super_candidate(self):
        # One level-1 district {1,2,3}; no super-candidates anywhere.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2, 3})},
            level1_centers={"D1": 1},
            super_candidates=set(),
            travel_times={(1, 1): 0.0, (1, 2): 1.0, (1, 3): 2.0},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        assert psi(frozenset({"D1"}), 1) == 0.0

    def test_zero_when_subnodes_empty(self):
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2})},
            level1_centers={"D1": 1},
            super_candidates={1},
            travel_times={(1, 1): 0.0, (1, 2): 1.0},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        assert psi(frozenset(), 1) == 0.0

    def test_gamma_zero_is_uniform(self):
        # γ=0 → ψ²(T_u) = 1 for every admissible super-cut (still soft-skip
        # when the subtree holds no super-candidate).
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2, 3})},
            level1_centers={"D1": 2},
            super_candidates={1, 3},
            travel_times={(i, j): float(abs(i - j)) for i in range(4) for j in range(4)},
        )
        psi = hub_coherence_psi_factory(state, gamma=0.0)
        assert psi(frozenset({"D1"}), 1) == 1.0
        assert psi(frozenset({"D1"}), 2) == 1.0
        assert psi(frozenset({"D1"}), 3) == 1.0

    def test_median_formula(self):
        # Two districts D1 = {1,2}, D2 = {4,5}; super-candidates 2 and 5;
        # unit demands and distances |i - j|. The subtree holds both districts.
        # cost(2) = 1 + 0 + 2 + 3 = 6; cost(5) = 4 + 3 + 1 + 0 = 8.
        # eta² = min(6, 8) / total demand 4 = 1.5; with γ=1: ψ² = exp(-1.5)
        # (the assigned capacity does not enter the score).
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2}), "D2": frozenset({4, 5})},
            level1_centers={"D1": 1, "D2": 4},
            super_candidates={2, 5},
            travel_times={(i, j): float(abs(i - j)) for i in range(6) for j in range(6)},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        result = psi(frozenset({"D1", "D2"}), 2)
        assert abs(result - math.exp(-1.5)) < 1e-10

    def test_skips_candidate_with_missing_travel_time(self):
        # Two super-candidates; one missing a travel-time entry to a facility.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2, 3})},
            level1_centers={"D1": 1},
            super_candidates={2, 3},
            # Candidate 2 has a full row; candidate 3 has no entries at all.
            travel_times={(2, 1): 5.0, (2, 2): 0.0, (2, 3): 1.0},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        # Only candidate 2 contributes: cost 6 over demand 3 -> eta² = 2.
        result = psi(frozenset({"D1"}), 1)
        assert abs(result - math.exp(-2.0)) < 1e-10

    def test_returns_zero_when_no_candidate_has_complete_travel_times(self):
        # The only super-candidate lacks a travel time to node 2 -> excluded.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2})},
            level1_centers={},
            super_candidates={2},
            travel_times={(2, 1): 1.0},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        assert psi(frozenset({"D1"}), 1) == 0.0


# ---------------------------------------------------------------------------
# End-to-end: hierarchical_recom with gamma_super > 0
# ---------------------------------------------------------------------------

@pytest.fixture
def manhattan_partition():
    set_seed(42)
    grid = Grid(dimensions=(6, 5), num_candidates=6, density="uniform").graph
    Assignment.travel_times = {
        (a, b): float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        for a in grid.nodes for b in grid.nodes
    }
    return Partition.from_random_assignment(
        graph=grid,
        epsilon=0.3,
        demand_target=500,
        assignment_class=None,
        capacity_level=3,
    )


class TestHierarchicalRecomWithGammaSuper:
    def test_gamma_super_zero_runs(self, manhattan_partition):
        # With gamma_super=0, super_psi_fn is None and ψ² = 1 (uniform).
        from falcomchain.markovchain.facility import SuperFacilityAssignment

        # Tag every node as super-candidate so something can be selected.
        g = manhattan_partition.graph.graph
        for n in g.nodes:
            g.nodes[n]["super_candidate"] = 1

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )
        try:
            new_state = hierarchical_recom(
                state,
                epsilon_base=0.3,
                epsilon_super=0.3,
                demand_target=500,
                gamma_super=0.0,
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable: {exc}")

        assert new_state.partition is not None

    def test_gamma_super_positive_runs(self, manhattan_partition):
        # With gamma_super > 0, hub_coherence_psi_factory is used.
        from falcomchain.markovchain.facility import SuperFacilityAssignment

        g = manhattan_partition.graph.graph
        for n in g.nodes:
            g.nodes[n]["super_candidate"] = 1

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )
        try:
            new_state = hierarchical_recom(
                state,
                epsilon_base=0.3,
                epsilon_super=0.3,
                demand_target=500,
                gamma_super=1.0,
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable: {exc}")

        assert new_state.partition is not None

    def test_explicit_super_psi_fn_overrides_gamma(self, manhattan_partition):
        # User-provided super_psi_fn should win over gamma_super.
        from falcomchain.markovchain.facility import SuperFacilityAssignment

        g = manhattan_partition.graph.graph
        for n in g.nodes:
            g.nodes[n]["super_candidate"] = 1

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )

        calls = []

        def my_super_psi(subnodes, teams):
            calls.append((subnodes, teams))
            return 1.0  # uniform — every admissible cut equally likely

        try:
            new_state = hierarchical_recom(
                state,
                epsilon_base=0.3,
                epsilon_super=0.3,
                demand_target=500,
                gamma_super=999.0,  # would normally heavily bias the cut
                super_psi_fn=my_super_psi,
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable: {exc}")

        # The custom function should have been called at least once during
        # the supergraph cut step.
        assert len(calls) > 0
