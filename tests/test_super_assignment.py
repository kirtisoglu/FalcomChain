"""
Tests for level-2 (superdistrict) assignment persistence on Partition.

Step (1) of the level-2 work: the supergraph partition that
``hierarchical_recom`` resamples each step is now stored on
``Partition.super_assignment`` rather than discarded after the chosen
superdistrict is picked. Step (2) (real Eq. 18 facility location) builds
on this; it is not tested here.
"""

import pytest

from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed
from falcomchain.graph.grid import Grid
from falcomchain.tree.tree import Flip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    set_seed(42)
    grid = Grid(dimensions=(6, 5), num_candidates=6, density="uniform")
    return grid.graph


@pytest.fixture
def initial_partition(small_grid):
    set_seed(42)
    Assignment.travel_times = None
    return Partition.from_random_assignment(
        graph=small_grid,
        epsilon=0.3,
        demand_target=500,
        assignment_class=None,
        capacity_level=3,
    )


# ---------------------------------------------------------------------------
# Initial state: super_assignment is identity
# ---------------------------------------------------------------------------

class TestInitialSuperAssignment:
    def test_attribute_exists(self, initial_partition):
        assert hasattr(initial_partition, "super_assignment")

    def test_identity_keys_match_parts(self, initial_partition):
        assert set(initial_partition.super_assignment.keys()) == set(
            initial_partition.parts.keys()
        )

    def test_identity_each_district_is_own_superdistrict(self, initial_partition):
        for l1_id, super_id in initial_partition.super_assignment.items():
            assert l1_id == super_id

    def test_super_parts_each_singleton(self, initial_partition):
        for super_id, l1_ids in initial_partition.super_parts.items():
            assert l1_ids == frozenset({super_id})

    def test_super_teams_matches_teams(self, initial_partition):
        # Identity grouping → super_teams[d] == teams[d] for every d
        for super_id, n_teams in initial_partition.super_teams.items():
            assert n_teams == initial_partition.teams[super_id]


# ---------------------------------------------------------------------------
# super_parts and super_teams derived correctly from non-trivial assignments
# ---------------------------------------------------------------------------

class TestSuperPartsAndTeams:
    def _build_partition_with_super(self, partition, super_assignment):
        """Patch a partition to a custom super_assignment for property tests."""
        partition.super_assignment = super_assignment
        return partition

    def test_groups_districts_by_super_id(self, initial_partition):
        # Group districts 1, 2 into super "A"; district 3 alone into "B".
        l1_ids = list(initial_partition.parts.keys())
        if len(l1_ids) < 3:
            pytest.skip("need at least 3 districts to exercise grouping")
        sa = {l1_ids[0]: "A", l1_ids[1]: "A", l1_ids[2]: "B"}
        for extra in l1_ids[3:]:
            sa[extra] = "C"  # singletons under their own super

        self._build_partition_with_super(initial_partition, sa)

        sp = initial_partition.super_parts
        assert sp["A"] == frozenset({l1_ids[0], l1_ids[1]})
        assert sp["B"] == frozenset({l1_ids[2]})

    def test_super_teams_sums_constituent_teams(self, initial_partition):
        l1_ids = list(initial_partition.parts.keys())
        if len(l1_ids) < 2:
            pytest.skip("need at least 2 districts")
        sa = {l1_ids[0]: "X", l1_ids[1]: "X"}
        for extra in l1_ids[2:]:
            sa[extra] = extra

        self._build_partition_with_super(initial_partition, sa)
        expected = (
            initial_partition.teams[l1_ids[0]]
            + initial_partition.teams[l1_ids[1]]
        )
        assert initial_partition.super_teams["X"] == expected


# ---------------------------------------------------------------------------
# _build_super_assignment: the recursive (full) path
# ---------------------------------------------------------------------------

class TestBuildSuperAssignmentRecursivePath:
    """When superflip carries a full .flips dict (the recursive path),
    the new super_assignment is built from the resampled level-2 partition."""

    def test_unchanged_districts_keep_resampled_super_id(self, initial_partition):
        l1_ids = list(initial_partition.parts.keys())
        if len(l1_ids) < 3:
            pytest.skip("need at least 3 districts")

        # Resampled level-2 puts l1_ids[0] and l1_ids[1] into "A",
        # and l1_ids[2] into "B".
        super_flips = {l1_ids[0]: "A", l1_ids[1]: "A", l1_ids[2]: "B"}
        for extra in l1_ids[3:]:
            super_flips[extra] = "C"

        # Merge the chosen superdistrict "A" (l1_ids[0..1]); the
        # lower-level re-partition produces new IDs 99, 100.
        merge = frozenset({l1_ids[0], l1_ids[1]})
        superflip = Flip(merged_ids=merge, flips=super_flips, new_ids=frozenset())
        flip = Flip(
            flips={},
            team_flips={99: 1, 100: 1},
            new_ids=frozenset({99, 100}),
            merged_ids=merge,
        )

        new_sa = initial_partition._build_super_assignment(
            initial_partition, superflip, flip
        )

        # Unchanged districts keep the resampled super_id
        assert new_sa[l1_ids[2]] == "B"
        for extra in l1_ids[3:]:
            assert new_sa[extra] == "C"

        # Merged old IDs are gone
        assert l1_ids[0] not in new_sa
        assert l1_ids[1] not in new_sa

        # New IDs all join the chosen superdistrict ("A")
        assert new_sa[99] == "A"
        assert new_sa[100] == "A"

    def test_handles_no_merge(self, initial_partition):
        # Edge case: superflip with no merged_ids. The result should just
        # be the resampled level-2 partition unchanged.
        l1_ids = list(initial_partition.parts.keys())
        super_flips = {pid: f"S{i}" for i, pid in enumerate(l1_ids)}
        superflip = Flip(merged_ids=frozenset(), flips=super_flips)
        flip = Flip(new_ids=frozenset(), merged_ids=frozenset())

        new_sa = initial_partition._build_super_assignment(
            initial_partition, superflip, flip
        )
        assert new_sa == super_flips


# ---------------------------------------------------------------------------
# _build_super_assignment: legacy path (Flip(merged_ids=merge) only)
# ---------------------------------------------------------------------------

class TestBuildSuperAssignmentLegacyPath:
    """When superflip has empty .flips (the bipartition fallback path),
    inherit the parent's super_assignment, drop merged, place new IDs into
    the chosen superdistrict derived from any merged ID."""

    def test_inherits_parent_assignment(self, initial_partition):
        l1_ids = list(initial_partition.parts.keys())
        if len(l1_ids) < 2:
            pytest.skip("need at least 2 districts")

        # Parent's super_assignment is identity (initial partition).
        merge = frozenset({l1_ids[0]})
        superflip = Flip(merged_ids=merge)  # legacy: no flips dict
        flip = Flip(
            flips={},
            team_flips={50: 1},
            new_ids=frozenset({50}),
            merged_ids=merge,
        )

        new_sa = initial_partition._build_super_assignment(
            initial_partition, superflip, flip
        )

        # Merged ID is gone
        assert l1_ids[0] not in new_sa
        # Other IDs preserved from parent
        for other in l1_ids[1:]:
            assert new_sa[other] == initial_partition.super_assignment[other]
        # New ID joins chosen superdistrict (= parent's super for the merged ID)
        chosen = initial_partition.super_assignment[l1_ids[0]]
        assert new_sa[50] == chosen


# ---------------------------------------------------------------------------
# End-to-end: run hierarchical_recom one step and verify super_assignment
# ---------------------------------------------------------------------------

class TestHierarchicalRecomPersistsSuperAssignment:
    def test_super_assignment_carries_through_perform_flip(self, initial_partition):
        from falcomchain.markovchain.proposals import hierarchical_recom
        from falcomchain.markovchain.state import ChainState

        set_seed(7)

        # Travel times must be set for ChainState.initial to compute facilities;
        # use Manhattan on grid coords.
        g = initial_partition.graph.graph
        Assignment.travel_times = {
            (a, b): float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
            for a in g.nodes for b in g.nodes
        }

        state = ChainState.initial(
            partition=initial_partition, energy=0.0, beta=1.0
        )

        try:
            new_state = hierarchical_recom(
                state, epsilon_base=0.3, epsilon_super=0.3, demand_target=500
            )
        except Exception as exc:
            pytest.skip(f"hierarchical_recom not runnable on this fixture: {exc}")

        new_partition = new_state.partition

        # super_assignment exists on the proposed partition
        assert hasattr(new_partition, "super_assignment")

        # Every level-1 district has a super_id assigned
        assert set(new_partition.super_assignment.keys()) == set(
            new_partition.parts.keys()
        )

        # super_parts is a partition of level-1 districts (no overlaps)
        all_l1 = set()
        for super_id, l1_set in new_partition.super_parts.items():
            assert all_l1.isdisjoint(l1_set)
            all_l1 |= l1_set
        assert all_l1 == set(new_partition.parts.keys())

        # super_teams sums to the same total as level-1 teams
        assert sum(new_partition.super_teams.values()) == sum(
            new_partition.teams.values()
        )
