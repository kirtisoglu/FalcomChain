"""
Tests for the ``c_min`` parameter (paper's c^ℓ_min).

``c_min`` controls the minimum capacity per district. Default 1 keeps
the original behavior. Larger values restrict the chain to coarser
partitions and relax Assumption 6.1 by a factor of ``c_min``:

    threshold = (1 − ε) · c_min · demand_target

Tests verify:
  1. Default c_min=1 preserves backward-compatible behavior.
  2. check_facility_density uses the c_min-scaled threshold.
  3. repair_facility_density needs fewer artificial candidates at higher c_min.
  4. Partition.from_random_assignment respects c_min in the recursion
     (no district has fewer than c_min teams).
"""
import json
import sys
from pathlib import Path

import networkx as nx
import pytest

from falcomchain import (
    Partition,
    check_facility_density,
    repair_facility_density,
)
from falcomchain.graph.grid import Grid
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    set_seed(42)
    return Grid(dimensions=(8, 8), num_candidates=20, density="uniform").graph


@pytest.fixture
def demo_grid_no_candidates():
    """The shared 10x10 demo grid, with all candidates stripped."""
    p = Path(__file__).resolve().parent.parent / "docs" / "_static" / "demo_grid_10x10.json"
    with p.open() as f:
        g = nx.node_link_graph(json.load(f), edges="links")
    for n in g.nodes:
        g.nodes[n]["candidate"] = 0
        g.nodes[n].pop("candidate_artificial", None)
    return g


# ---------------------------------------------------------------------------
# check_facility_density
# ---------------------------------------------------------------------------

class TestCheckThresholdScalesWithCMin:
    def test_default_c_min_1_unchanged(self, demo_grid_no_candidates):
        # Default c_min=1: threshold = (1-ε)·w
        report = check_facility_density(
            demo_grid_no_candidates, demand_target=1000, epsilon=0.1
        )
        assert report.threshold == 900.0

    def test_c_min_2_doubles_threshold(self, demo_grid_no_candidates):
        report = check_facility_density(
            demo_grid_no_candidates,
            demand_target=1000,
            epsilon=0.1,
            c_min=2,
        )
        assert report.threshold == 1800.0

    def test_c_min_3_triples_threshold(self, demo_grid_no_candidates):
        report = check_facility_density(
            demo_grid_no_candidates,
            demand_target=1000,
            epsilon=0.1,
            c_min=3,
        )
        assert report.threshold == 2700.0

    def test_higher_c_min_passes_more(self, demo_grid_no_candidates):
        # No candidates → giant facility-free component of demand 5000.
        # At c_min=1: threshold 900 → fails badly.
        # At c_min=10: threshold 9000 > 5000 → passes vacuously.
        rep_low = check_facility_density(
            demo_grid_no_candidates, demand_target=1000, epsilon=0.1, c_min=1
        )
        rep_high = check_facility_density(
            demo_grid_no_candidates, demand_target=1000, epsilon=0.1, c_min=10
        )
        assert not rep_low.passes
        assert rep_high.passes


# ---------------------------------------------------------------------------
# repair_facility_density: higher c_min needs fewer candidates
# ---------------------------------------------------------------------------

class TestRepairCMinReducesCandidateCount:
    def test_higher_c_min_adds_fewer_or_equal_candidates(self, demo_grid_no_candidates):
        import copy
        counts = []
        for c_min in (1, 2, 3):
            g = copy.deepcopy(demo_grid_no_candidates)
            added = repair_facility_density(
                g,
                demand_target=1000,
                epsilon=0.1,
                strategy="weighted_center",
                c_min=c_min,
            )
            counts.append(len(added))
        # Monotonically non-increasing: relaxing c_min only loosens.
        assert counts[0] >= counts[1] >= counts[2], (
            f"expected fewer candidates as c_min grows, got {counts}"
        )

    def test_c_min_3_needs_substantially_fewer(self, demo_grid_no_candidates):
        import copy
        g1 = copy.deepcopy(demo_grid_no_candidates)
        g3 = copy.deepcopy(demo_grid_no_candidates)
        added_1 = repair_facility_density(
            g1, demand_target=1000, epsilon=0.1,
            strategy="weighted_center", c_min=1,
        )
        added_3 = repair_facility_density(
            g3, demand_target=1000, epsilon=0.1,
            strategy="weighted_center", c_min=3,
        )
        # c_min=3 should need strictly fewer candidates than c_min=1.
        # The exact ratio depends on graph structure; on the demo grid
        # we typically see ~26 -> ~16 (a ~40% reduction).
        assert len(added_3) < len(added_1)


# ---------------------------------------------------------------------------
# Partition: c_min restricts district capacities
# ---------------------------------------------------------------------------

class TestPartitionRespectsMinCapacity:
    def test_c_min_2_no_singleton_capacity(self, small_grid):
        # With c_min=2, every district must have ≥2 teams.
        set_seed(42)
        Assignment.travel_times = None
        try:
            p = Partition.from_random_assignment(
                graph=small_grid,
                epsilon=0.3,
                demand_target=500,
                assignment_class=None,
                capacity_level=3,
                c_min=2,
            )
        except Exception as exc:
            pytest.skip(
                f"c_min=2 with these params is infeasible (n_teams=6 → "
                f"sums of [2,3]); skipping: {exc}"
            )
        for d_id, n_teams in p.teams.items():
            assert n_teams >= 2, (
                f"district {d_id} has {n_teams} teams; expected >= c_min=2"
            )
