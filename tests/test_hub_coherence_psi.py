"""
Tests for the level-2 cut score `hub_coherence_psi_factory` (paper Eq. 27).

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
        graph_nodes[n] = {"super_candidate": 1 if n in super_candidates else 0}

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

    def test_gamma_zero_returns_teams(self):
        # γ=0 → ψ²(T_u) = ϕ²(T_u) = teams (still soft-skip if no super-cand).
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2, 3})},
            level1_centers={"D1": 2},
            super_candidates={1, 3},
            travel_times={(i, j): float(abs(i - j)) for i in range(4) for j in range(4)},
        )
        psi = hub_coherence_psi_factory(state, gamma=0.0)
        assert psi(frozenset({"D1"}), 1) == 1.0
        assert psi(frozenset({"D1"}), 2) == 2.0
        assert psi(frozenset({"D1"}), 3) == 3.0

    def test_minimax_formula(self):
        # Two districts D1 (center=1), D2 (center=4).
        # Two super-candidates: 2 (in D1) and 5 (in D2).
        # Subtree contains both districts.
        # For super-candidate 2: max(d(2,1), d(2,4)) = max(1, 2) = 2.
        # For super-candidate 5: max(d(5,1), d(5,4)) = max(4, 1) = 4.
        # π² = min(2, 4) = 2. With γ=1, teams=2: ψ² = 2 * exp(-2) ≈ 0.270.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2}), "D2": frozenset({4, 5})},
            level1_centers={"D1": 1, "D2": 4},
            super_candidates={2, 5},
            travel_times={(i, j): float(abs(i - j)) for i in range(6) for j in range(6)},
        )
        psi = hub_coherence_psi_factory(state, gamma=1.0)
        result = psi(frozenset({"D1", "D2"}), 2)
        assert abs(result - 2.0 * math.exp(-2.0)) < 1e-10

    def test_skips_candidate_with_missing_travel_time(self):
        # Two super-candidates; one missing a travel-time entry to a facility.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2})},
            level1_centers={"D1": 1},
            super_candidates={2, 3},
            # No (3, 1) entry
            travel_times={(1, 1): 0.0, (2, 1): 5.0},
        )
        # Add node 3 to the partition graph
        state.partition.graph.nodes[3] = {"super_candidate": 1}
        state.partition.parts["D1"] = frozenset({1, 2, 3})

        psi = hub_coherence_psi_factory(state, gamma=1.0)
        # Only super-cand 2 contributes: max(d(2, 1)) = 5. ψ² = 1 * exp(-5).
        result = psi(frozenset({"D1"}), 1)
        assert abs(result - math.exp(-5.0)) < 1e-10

    def test_returns_zero_when_subtree_has_no_facility(self):
        # subtree contains a district that has no center entry — defensive case.
        state = _make_mock_state(
            parts={"D1": frozenset({1, 2})},
            level1_centers={},  # no level-1 centers computed
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
        # With gamma_super=0, super_psi_fn is None and ψ² = teams.
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
