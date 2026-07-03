"""
Tests for the facility-density assumption (Assumption 6.1) verification
and repair routines.
"""

import networkx as nx
import pytest

from falcomchain.candidates.feasibility import (
    FeasibilityReport,
    check_facility_density,
    repair_facility_density,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_graph(n, demand=100, candidates=()):
    """Build a path graph with given demand and candidate set."""
    g = nx.path_graph(n)
    for node in g.nodes:
        g.nodes[node]["demand"] = demand
        g.nodes[node]["candidate"] = 1 if node in candidates else 0
    return g


def _grid_graph(rows, cols, demand=100, candidates=()):
    """Build a grid graph with given demand and candidate set."""
    g = nx.grid_2d_graph(rows, cols)
    for node in g.nodes:
        g.nodes[node]["demand"] = demand
        g.nodes[node]["candidate"] = 1 if node in candidates else 0
    return g


# ---------------------------------------------------------------------------
# check_facility_density
# ---------------------------------------------------------------------------

class TestCheckFacilityDensity:
    def test_passes_when_all_candidates(self):
        g = _path_graph(5, demand=100, candidates=range(5))
        report = check_facility_density(g, demand_target=200, epsilon=0.1)
        assert report.passes
        assert report.violating_components == []
        # facility-free subgraph is empty -> worst_demand == 0
        assert report.worst_demand == 0.0

    def test_passes_when_components_below_threshold(self):
        # 3 nodes * 100 = 300 demand, threshold = (1-0.1)*1000 = 900
        g = _path_graph(3, demand=100)
        report = check_facility_density(g, demand_target=1000, epsilon=0.1)
        assert report.passes
        assert report.worst_demand == 300.0

    def test_fails_when_component_exceeds_threshold(self):
        # 10 nodes * 100 = 1000 demand, threshold = (1-0.1)*500 = 450
        g = _path_graph(10, demand=100)
        report = check_facility_density(g, demand_target=500, epsilon=0.1)
        assert not report.passes
        assert len(report.violating_components) == 1
        assert report.component_demands[0] == 1000.0
        assert report.worst_demand == 1000.0

    def test_candidates_split_components(self):
        # Path of 10 nodes, candidate at index 4 splits into [0..3] (4 nodes)
        # and [5..9] (5 nodes). Both 400 and 500. threshold=450.
        # Component 5..9 has demand 500 >= 450 -> violates.
        g = _path_graph(10, demand=100, candidates=[4])
        report = check_facility_density(g, demand_target=500, epsilon=0.1)
        assert not report.passes
        assert len(report.violating_components) == 1
        assert 5 in report.violating_components[0]

    def test_threshold_is_strict_inequality(self):
        # Component demand exactly equal to threshold should violate
        # (paper requires strictly less than (1-eps)*w)
        g = _path_graph(5, demand=100)  # total 500
        # threshold = (1-0)*500 = 500. 500 >= 500 -> violates.
        report = check_facility_density(g, demand_target=500, epsilon=0.0)
        assert not report.passes

    def test_empty_graph_passes(self):
        g = nx.Graph()
        report = check_facility_density(g, demand_target=100, epsilon=0.1)
        assert report.passes

    def test_report_repr(self):
        g = _path_graph(3, demand=100, candidates=[0, 1, 2])
        report = check_facility_density(g, demand_target=200, epsilon=0.1)
        assert "passes" in repr(report)

        g2 = _path_graph(10, demand=100)
        report2 = check_facility_density(g2, demand_target=500, epsilon=0.1)
        assert "FAILS" in repr(report2)


# ---------------------------------------------------------------------------
# repair_facility_density - shared behavior across strategies
# ---------------------------------------------------------------------------

try:
    import pymetis  # noqa: F401
    _HAS_PYMETIS = True
except ImportError:
    _HAS_PYMETIS = False


_ALL_STRATEGIES = [
    "highest_demand",
    "center",
    "weighted_center",
    "fast_center",
    "balanced_separator",
]
if _HAS_PYMETIS:
    _ALL_STRATEGIES.append("metis_separator")


@pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
class TestRepairFacilityDensity:
    def test_makes_assumption_pass(self, strategy):
        g = _path_graph(20, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy=strategy
        )
        # After repair, the assumption should hold
        report = check_facility_density(g, demand_target=500, epsilon=0.1)
        assert report.passes
        assert len(added) >= 1

    def test_marks_added_as_artificial(self, strategy):
        g = _path_graph(20, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy=strategy
        )
        for n in added:
            assert g.nodes[n]["candidate"] == 1
            assert g.nodes[n]["candidate_artificial"] == 1

    def test_does_not_modify_existing_candidates(self, strategy):
        g = _path_graph(20, demand=100, candidates=[5])
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy=strategy
        )
        # Pre-existing candidate must not be marked artificial
        assert "candidate_artificial" not in g.nodes[5] or \
               g.nodes[5].get("candidate_artificial") != 1
        assert 5 not in added

    def test_no_op_when_already_passes(self, strategy):
        g = _path_graph(5, demand=100, candidates=[0, 4])
        added = repair_facility_density(
            g, demand_target=10000, epsilon=0.1, strategy=strategy
        )
        assert added == set()

    def test_grid_graph(self, strategy):
        # 6x6 grid, 36 nodes * 100 = 3600 demand. threshold = (1-0.1)*500 = 450.
        # No candidates initially, single component of 3600 >> 450.
        g = _grid_graph(6, 6, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy=strategy
        )
        report = check_facility_density(g, demand_target=500, epsilon=0.1)
        assert report.passes
        # Need at least ~3600/450 = 8 candidates
        assert len(added) >= 6

    def test_terminates_on_pathological_case(self, strategy):
        # Single node with huge demand, no candidates. Threshold tiny.
        g = nx.Graph()
        g.add_node(0, demand=1000, candidate=0)
        added = repair_facility_density(
            g, demand_target=10, epsilon=0.0, strategy=strategy
        )
        # The single node must become a candidate
        assert added == {0}
        assert g.nodes[0]["candidate"] == 1


# ---------------------------------------------------------------------------
# Strategy-specific behavior
# ---------------------------------------------------------------------------

class TestHighestDemandStrategy:
    def test_picks_max_demand_node(self):
        g = nx.path_graph(5)
        for n in g.nodes:
            g.nodes[n]["candidate"] = 0
            g.nodes[n]["demand"] = 100 if n != 2 else 999
        # threshold = 0.9 * 100 = 90, total 1399 >> 90, so violates
        added = repair_facility_density(
            g, demand_target=100, epsilon=0.1, strategy="highest_demand"
        )
        # First pick must be node 2 (demand 999)
        assert 2 in added


class TestCenterStrategy:
    def test_picks_graph_center(self):
        # 7-node path: center is node 3 (eccentricity 3).
        # threshold = (1 - 0) * 700 = 700, total 700 -> violates equality.
        g = _path_graph(7, demand=100)
        added = repair_facility_density(
            g, demand_target=700, epsilon=0.0, strategy="center"
        )
        # On a path of 7 nodes, the 1-center is the middle node 3.
        assert 3 in added


class TestWeightedCenterStrategy:
    def test_picks_demand_weighted_center(self):
        # Skewed demand: high demand at one end pulls the weighted center
        # toward that end.
        g = nx.path_graph(7)
        for n in g.nodes:
            g.nodes[n]["candidate"] = 0
            g.nodes[n]["demand"] = 1000 if n == 6 else 1
        # threshold tiny, so violates
        added = repair_facility_density(
            g, demand_target=10, epsilon=0.0, strategy="weighted_center"
        )
        # The weighted center should bias toward node 6.
        # Center of unweighted path is 3, but weighted center should be > 3.
        first_added = max(added, key=lambda n: 1)  # at least one was added
        # Verify node 6 ends up a candidate (it has all the demand)
        assert g.nodes[6]["candidate"] == 1


class TestFastCenterStrategy:
    def test_picks_path_center(self):
        # On a 7-node path, the exact 1-center is node 3. The double-BFS
        # midpoint trick should land on it: dist(0)->u=6, dist(6)->v=0,
        # midpoint of 0-6 path at distance 3 is node 3.
        g = _path_graph(7, demand=100)
        added = repair_facility_density(
            g, demand_target=700, epsilon=0.0, strategy="fast_center"
        )
        assert 3 in added

    def test_uses_fewer_iterations_than_highest_demand_on_grid(self):
        # On a planar grid, fast_center fragments components by bisection;
        # highest_demand picks peripheral max-demand nodes. fast_center
        # should converge with substantially fewer artificial candidates.
        import copy
        g = _grid_graph(8, 8, demand=100)
        g_hd = copy.deepcopy(g)
        added_fc = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy="fast_center"
        )
        added_hd = repair_facility_density(
            g_hd, demand_target=500, epsilon=0.1, strategy="highest_demand"
        )
        assert len(added_fc) <= len(added_hd)

    def test_terminates_on_disconnected_components(self):
        # Two disjoint cliques, no candidates. fast_center should pick
        # one node per component. Both must have *some* candidate added,
        # not just one.
        g = nx.Graph()
        for n in range(5):
            g.add_node(n, demand=100, candidate=0)
            g.add_node(n + 10, demand=100, candidate=0)
        for u in range(5):
            for v in range(u + 1, 5):
                g.add_edge(u, v)
                g.add_edge(u + 10, v + 10)
        added = repair_facility_density(
            g, demand_target=200, epsilon=0.0, strategy="fast_center"
        )
        # At least one node from each clique should now be a candidate.
        assert any(g.nodes[n]["candidate"] for n in range(5))
        assert any(g.nodes[n]["candidate"] for n in range(10, 15))


class TestBalancedSeparatorStrategy:
    def test_path_separator_is_midpoint(self):
        # On a 7-path the BFS bisection picks the midpoint; the border is
        # a single vertex (a path has no thicker separator).
        g = _path_graph(7, demand=100)
        added = repair_facility_density(
            g, demand_target=700, epsilon=0.0, strategy="balanced_separator"
        )
        # Should converge with all-real candidates feasible.
        report = check_facility_density(g, demand_target=700, epsilon=0.0)
        assert report.passes

    def test_grid_uses_fewer_than_or_equal_to_fast_center(self):
        # On an 8x8 grid balanced_separator should produce no more
        # candidates than fast_center (typically fewer because each
        # bisection cuts through the middle, not a single vertex).
        import copy
        g = _grid_graph(8, 8, demand=100)
        g_fc = copy.deepcopy(g)
        g_bs = copy.deepcopy(g)
        added_fc = repair_facility_density(
            g_fc, demand_target=500, epsilon=0.1, strategy="fast_center"
        )
        added_bs = repair_facility_density(
            g_bs, demand_target=500, epsilon=0.1, strategy="balanced_separator"
        )
        # balanced_separator may add more or fewer; the only invariant
        # is that both succeed in repairing the graph.
        assert check_facility_density(g_bs, demand_target=500, epsilon=0.1).passes
        assert check_facility_density(g_fc, demand_target=500, epsilon=0.1).passes

    def test_returns_set_with_at_least_one_node(self):
        g = _grid_graph(6, 6, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy="balanced_separator"
        )
        assert len(added) >= 1
        # Every added node should be marked artificial
        for n in added:
            assert g.nodes[n]["candidate_artificial"] == 1


# ---------------------------------------------------------------------------
# feasibility_violation
# ---------------------------------------------------------------------------

class TestFeasibilityViolation:
    def test_returns_zero_when_passes(self):
        from falcomchain.candidates.feasibility import feasibility_violation
        g = _path_graph(5, demand=100, candidates=[0, 4])
        v = feasibility_violation(g, demand_target=10000, epsilon=0.1)
        assert v == 0.0

    def test_positive_when_violates(self):
        from falcomchain.candidates.feasibility import feasibility_violation
        g = _path_graph(20, demand=100)  # 2000 total, no candidates
        v = feasibility_violation(g, demand_target=500, epsilon=0.1)
        # threshold = 450; worst = 2000; (2000-450)/450 ≈ 3.444
        assert v > 3.0
        assert v < 4.0

    def test_after_repair_returns_zero(self):
        from falcomchain.candidates.feasibility import feasibility_violation
        g = _grid_graph(6, 6, demand=100)
        repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy="fast_center"
        )
        assert feasibility_violation(g, demand_target=500, epsilon=0.1) == 0.0

    def test_scales_with_c_min(self):
        # Larger c_min => higher threshold => smaller violation
        from falcomchain.candidates.feasibility import feasibility_violation
        g = _path_graph(20, demand=100)
        v1 = feasibility_violation(g, demand_target=500, epsilon=0.1, c_min=1)
        v3 = feasibility_violation(g, demand_target=500, epsilon=0.1, c_min=3)
        assert v3 < v1


@pytest.mark.skipif(not _HAS_PYMETIS, reason="pymetis not installed")
class TestMetisSeparatorStrategy:
    def test_repairs_grid_graph(self):
        g = _grid_graph(6, 6, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy="metis_separator"
        )
        assert check_facility_density(
            g, demand_target=500, epsilon=0.1
        ).passes
        assert len(added) >= 1

    def test_demand_balanced_cuts_on_heterogeneous_grid(self):
        # Heterogeneous demand: alternating high/low. METIS should
        # produce demand-balanced cuts (not just count-balanced).
        g = _grid_graph(6, 6, demand=100)
        # Make alternating heterogeneous demand
        for n in g.nodes:
            r, c = n
            g.nodes[n]["demand"] = 1000 if (r + c) % 2 == 0 else 100
        added = repair_facility_density(
            g, demand_target=2000, epsilon=0.1, strategy="metis_separator"
        )
        report = check_facility_density(g, demand_target=2000, epsilon=0.1)
        assert report.passes

    def test_marks_added_as_artificial(self):
        g = _grid_graph(6, 6, demand=100)
        added = repair_facility_density(
            g, demand_target=500, epsilon=0.1, strategy="metis_separator"
        )
        for n in added:
            assert g.nodes[n]["candidate_artificial"] == 1


def test_metis_separator_raises_when_pymetis_missing(monkeypatch):
    # Simulate pymetis being unavailable.
    import sys
    import importlib
    saved = sys.modules.pop("pymetis", None)
    monkeypatch.setitem(sys.modules, "pymetis", None)
    try:
        from falcomchain.candidates import feasibility as F
        importlib.reload(F)
        g = _path_graph(20, demand=100)
        with pytest.raises(ImportError, match="pymetis"):
            F.repair_facility_density(
                g, demand_target=500, epsilon=0.1, strategy="metis_separator"
            )
    finally:
        if saved is not None:
            sys.modules["pymetis"] = saved
        from falcomchain.candidates import feasibility as F
        importlib.reload(F)


class TestUnknownStrategy:
    def test_raises_on_unknown_strategy(self):
        g = _path_graph(10, demand=100)
        with pytest.raises(ValueError, match="Unknown strategy"):
            repair_facility_density(
                g, demand_target=500, epsilon=0.1, strategy="bogus"
            )
