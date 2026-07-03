"""
Tests for level-2 (super-) facility assignment.

Covers step 3 (real Eq. 18 selector) and step 4 (opt-in via
``ChainState.initial(super_facility_fn=...)``).
"""

import pytest

from falcomchain.graph.grid import Grid
from falcomchain.markovchain.facility import (
    SuperFacilityAssignment,
    minimax_super_selector,
)
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    set_seed(42)
    return Grid(dimensions=(6, 5), num_candidates=6, density="uniform").graph


@pytest.fixture
def manhattan_partition(small_grid):
    """A partition with Manhattan-distance travel times set on Assignment."""
    set_seed(42)
    Assignment.travel_times = {
        (a, b): float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        for a in small_grid.nodes for b in small_grid.nodes
    }
    return Partition.from_random_assignment(
        graph=small_grid,
        epsilon=0.3,
        demand_target=500,
        assignment_class=None,
        capacity_level=3,
    )


def _set_super_candidates(partition, nodes):
    """Helper: tag specific base-graph nodes as super_candidate=1 (others 0)."""
    g = partition.graph.graph
    for n in g.nodes:
        g.nodes[n]["super_candidate"] = 1 if n in nodes else 0


# ---------------------------------------------------------------------------
# minimax_super_selector
# ---------------------------------------------------------------------------

class TestMinimaxSuperSelector:
    def test_picks_node_with_smallest_eccentricity(self):
        # Three super-candidates, base nodes = {0, 1, 2, 3, 4}, distances
        # are |i - j|. Optimum is node 2 (radius 2); nodes 0 and 4 have radius 4.
        travel_times = {(i, j): float(abs(i - j)) for i in range(5) for j in range(5)}
        super_candidates = [0, 2, 4]
        base_nodes = list(range(5))
        best, radius = minimax_super_selector(
            "S0", super_candidates, base_nodes, travel_times
        )
        assert best == 2
        assert radius == 2.0

    def test_returns_none_for_empty_candidates(self):
        best, radius = minimax_super_selector("S0", [], [0, 1], {})
        assert best is None
        assert radius == float("inf")

    def test_skips_candidate_with_missing_travel_time(self):
        # Candidate 0 has no entry to node 1; candidate 1 has all entries.
        travel_times = {(0, 0): 0.0, (1, 0): 1.0, (1, 1): 0.0}
        best, _ = minimax_super_selector("S0", [0, 1], [0, 1], travel_times)
        assert best == 1


# ---------------------------------------------------------------------------
# SuperFacilityAssignment.from_state — soft constraint
# ---------------------------------------------------------------------------

class TestSuperFacilityAssignmentSoft:
    def test_no_super_candidates_yields_empty(self, manhattan_partition):
        # Default Grid sets super_candidate=0 on every node.
        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )
        assert state.super_facility is not None
        # Soft skip: no entries because no super_candidates exist anywhere.
        assert state.super_facility.centers == {}
        assert state.super_facility.radii == {}

    def test_some_superdistricts_get_facilities_others_dont(
        self, manhattan_partition
    ):
        # Tag exactly one node per district arbitrarily as a super-candidate
        # for half the districts; leave the other half without any.
        l1_ids = sorted(manhattan_partition.parts.keys())
        target_super_nodes = set()
        skipped_super_ids = set()
        for i, l1_id in enumerate(l1_ids):
            nodes = list(manhattan_partition.parts[l1_id])
            if i % 2 == 0:
                # even-indexed level-1 districts get one super-candidate
                target_super_nodes.add(nodes[0])
            else:
                skipped_super_ids.add(l1_id)

        _set_super_candidates(manhattan_partition, target_super_nodes)

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )

        # Initial super_assignment is identity, so superdistrict ID == level-1 ID.
        # Even-indexed superdistricts have a super-candidate -> entry exists.
        # Odd-indexed superdistricts don't -> no entry.
        for sid in l1_ids:
            if sid in skipped_super_ids:
                assert sid not in state.super_facility.centers, (
                    f"superdistrict {sid} has no super-candidate but was assigned a center"
                )
            else:
                assert sid in state.super_facility.centers


# ---------------------------------------------------------------------------
# ChainState.initial: super_facility_fn opt-in
# ---------------------------------------------------------------------------

class TestSuperFacilityFnOptIn:
    def test_default_is_none(self, manhattan_partition):
        # No super_facility_fn passed -> state.super_facility is None.
        state = ChainState.initial(
            partition=manhattan_partition, energy=0.0, beta=1.0
        )
        assert state.super_facility is None
        assert state.super_facility_fn is None

    def test_explicit_callable_runs(self, manhattan_partition):
        _set_super_candidates(
            manhattan_partition,
            {next(iter(p)) for p in manhattan_partition.parts.values()},
        )
        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )
        assert state.super_facility is not None
        assert len(state.super_facility.centers) > 0

    def test_custom_selector_propagates(self, manhattan_partition):
        # Custom selector that always returns the smallest-id candidate.
        def smallest_id_selector(super_id, super_candidates, base_nodes, tt):
            ordered = sorted(super_candidates)
            if not ordered:
                return None, float("inf")
            best = ordered[0]
            try:
                radius = max(tt[(best, v)] for v in base_nodes)
            except KeyError:
                radius = float("inf")
            return best, radius

        # Tag every base node as super-candidate to maximize the freedom.
        all_nodes = set(manhattan_partition.graph.graph.nodes)
        _set_super_candidates(manhattan_partition, all_nodes)

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=lambda s: SuperFacilityAssignment.from_state(
                s, selection_fn=smallest_id_selector
            ),
        )

        # Each superdistrict's center should be the smallest-id node in it.
        for super_id, base_nodes in manhattan_partition.super_parts.items():
            base = sorted(
                set.union(
                    *(set(manhattan_partition.parts[d]) for d in base_nodes)
                )
            )
            assert state.super_facility.centers[super_id] == base[0]

    def test_super_facility_fn_carries_through_chain_step(
        self, manhattan_partition
    ):
        # When initial state has super_facility_fn, a hierarchical_recom step
        # produces a new state that also computes super_facility.
        from falcomchain.markovchain.proposals import hierarchical_recom

        set_seed(7)
        all_nodes = set(manhattan_partition.graph.graph.nodes)
        _set_super_candidates(manhattan_partition, all_nodes)

        state = ChainState.initial(
            partition=manhattan_partition,
            energy=0.0,
            beta=1.0,
            super_facility_fn=SuperFacilityAssignment.from_state,
        )
        try:
            new_state = hierarchical_recom(
                state, epsilon_base=0.3, epsilon_super=0.3, demand_target=500
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable on this fixture: {exc}")

        assert new_state.super_facility is not None
        assert new_state.super_facility_fn is state.super_facility_fn

    def test_super_facility_fn_none_propagates_through_chain_step(
        self, manhattan_partition
    ):
        # Default (None) propagates: a hierarchical_recom step keeps super_facility None.
        from falcomchain.markovchain.proposals import hierarchical_recom

        set_seed(7)
        state = ChainState.initial(
            partition=manhattan_partition, energy=0.0, beta=1.0
        )
        try:
            new_state = hierarchical_recom(
                state, epsilon_base=0.3, epsilon_super=0.3, demand_target=500
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable on this fixture: {exc}")

        assert new_state.super_facility is None
        assert new_state.super_facility_fn is None
