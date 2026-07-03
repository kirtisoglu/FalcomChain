"""
Tests for the ensemble analysis module.

Uses the same small synthetic grid (6x5 = 30 nodes) as test_pipeline.py
to verify boundary frequency, facility stability, and capacity stats.
"""

import pytest

from falcomchain.random import set_seed
from falcomchain.graph.grid import Grid
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.chain import MarkovChain
from falcomchain.markovchain.facility import FacilityAssignment

from falcomchain.ensemble import (
    BoundaryCounter,
    CapacityStats,
    EnsembleStats,
    FacilityCounter,
    WelfordStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """6x5 grid with 30 nodes, demand=100 each, ~6 candidates."""
    set_seed(42)
    grid = Grid(dimensions=(6, 5), num_candidates=6, density="uniform")
    return grid.graph


def _make_state(small_grid, with_travel_times=True):
    """Build a ChainState with travel times set up."""
    partition = Partition.from_random_assignment(
        graph=small_grid,
        epsilon=0.3,
        demand_target=500,
        assignment_class=None,
        capacity_level=3,
    )

    if with_travel_times:
        travel_times = {}
        g = partition.graph.graph
        for n1 in g.nodes:
            for n2 in g.nodes:
                d = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])
                travel_times[(n1, n2)] = float(d)
        Assignment.travel_times = travel_times
        state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    else:
        Assignment.travel_times = None
        state = ChainState.__new__(ChainState)
        state.partition = partition
        state.energy = 0.0
        state.beta = 0.0
        state.log_proposal_ratio = 0.0
        state.feasible = True
        state.energy_fn = None
        state.super_facility = None
        state._recorder = None

        class FakeFacility:
            centers = {}
            radii = {}
            def center(self, p): return None
            def radius(self, p): return float("inf")
        state.facility = FakeFacility()

    return state


# ---------------------------------------------------------------------------
# WelfordStats tests
# ---------------------------------------------------------------------------

class TestWelfordStats:
    def test_empty(self):
        w = WelfordStats()
        assert w.n == 0
        assert w.mean == 0.0
        assert w.min is None

    def test_single_value(self):
        w = WelfordStats()
        w.update(5.0)
        assert w.n == 1
        assert w.mean == 5.0
        assert w.min == 5.0
        assert w.max == 5.0
        assert w.variance == 0.0

    def test_known_sequence(self):
        w = WelfordStats()
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            w.update(x)
        assert w.n == 8
        assert abs(w.mean - 5.0) < 1e-10
        assert w.min == 2.0
        assert w.max == 9.0
        # Sample variance of [2,4,4,4,5,5,7,9] = 4.571...
        assert abs(w.variance - 32 / 7) < 1e-10

    def test_summary(self):
        w = WelfordStats()
        w.update(3.0)
        w.update(7.0)
        s = w.summary()
        assert s["n"] == 2
        assert abs(s["mean"] - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# BoundaryCounter tests
# ---------------------------------------------------------------------------

class TestBoundaryCounter:
    def test_single_observation(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid)
        bc = BoundaryCounter()
        bc.observe(state, accepted=True)

        assert bc.n_samples == 1
        freqs = bc.frequencies()
        assert len(freqs) > 0
        # All frequencies should be 1.0 after one sample
        for f in freqs.values():
            assert f == 1.0

    def test_rejected_step_recounts(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid)
        bc = BoundaryCounter()
        bc.observe(state, accepted=True)
        bc.observe(state, accepted=False)  # same state, re-counted

        assert bc.n_samples == 2
        for f in bc.frequencies().values():
            assert f == 1.0  # same boundaries both times

    def test_frequencies_bounded(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid)
        bc = BoundaryCounter()
        bc.observe(state, accepted=True)

        for f in bc.frequencies().values():
            assert 0.0 <= f <= 1.0

    def test_robust_and_fragile(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid)
        bc = BoundaryCounter()
        bc.observe(state, accepted=True)

        # After one sample, all are 100% -> all robust, none fragile
        robust = bc.robust(threshold=0.9)
        fragile = bc.fragile(threshold=0.5)
        assert len(robust) == len(bc.frequencies())
        assert len(fragile) == 0

    def test_empty(self):
        bc = BoundaryCounter()
        assert bc.n_samples == 0
        assert bc.frequencies() == {}


# ---------------------------------------------------------------------------
# FacilityCounter tests
# ---------------------------------------------------------------------------

class TestFacilityCounter:
    def test_single_observation(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        fc = FacilityCounter()
        fc.observe(state, accepted=True)

        assert fc.n_samples == 1
        freqs = fc.frequencies()
        # Should have one center per district
        assert len(freqs) == len(state.partition.parts)
        for f in freqs.values():
            assert f == 1.0

    def test_centers_are_candidates(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        fc = FacilityCounter()
        fc.observe(state, accepted=True)

        # Every counted node should be a candidate
        all_candidates = set()
        for cands in state.partition.candidates.values():
            all_candidates |= cands
        for node in fc.frequencies():
            assert node in all_candidates

    def test_essential_and_substitutable(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        fc = FacilityCounter()
        fc.observe(state, accepted=True)

        essential = fc.essential(threshold=0.9)
        substitutable = fc.substitutable(threshold=0.5)
        assert len(essential) == len(fc.frequencies())
        assert len(substitutable) == 0

    def test_empty(self):
        fc = FacilityCounter()
        assert fc.n_samples == 0
        assert fc.frequencies() == {}
        assert fc.super_frequencies() == {}


# ---------------------------------------------------------------------------
# CapacityStats tests
# ---------------------------------------------------------------------------

class TestCapacityStats:
    def test_single_observation(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        cs = CapacityStats()
        cs.observe(state, accepted=True)

        assert cs.n_samples == 1
        assert cs.demand_cv.n == 1
        assert cs.max_radius.n == 1
        assert cs.mean_radius.n == 1

    def test_demand_cv_nonnegative(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        cs = CapacityStats()
        cs.observe(state, accepted=True)
        assert cs.demand_cv.mean >= 0.0

    def test_radii_positive(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        cs = CapacityStats()
        cs.observe(state, accepted=True)
        assert cs.max_radius.mean >= cs.mean_radius.mean

    def test_summary_keys(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        cs = CapacityStats()
        cs.observe(state, accepted=True)
        s = cs.summary()
        assert "demand_cv" in s
        assert "max_radius" in s
        assert "mean_radius" in s


# ---------------------------------------------------------------------------
# EnsembleStats tests
# ---------------------------------------------------------------------------

class TestEnsembleStats:
    def test_observe_delegates(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        es = EnsembleStats()
        es.observe(state, accepted=True)

        assert es.n_samples == 1
        assert es.boundary.n_samples == 1
        assert es.facility.n_samples == 1
        assert es.capacity.n_samples == 1

    def test_burn_in(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        es = EnsembleStats(burn_in=3)

        for _ in range(5):
            es.observe(state, accepted=True)

        # 5 steps - 3 burn-in = 2 recorded
        assert es.n_samples == 2

    def test_thinning(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        es = EnsembleStats(thin=2)

        for _ in range(6):
            es.observe(state, accepted=True)

        # Steps 1,2,3,4,5,6 -> record at 1,3,5 = 3 samples
        assert es.n_samples == 3

    def test_burn_in_and_thinning(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        es = EnsembleStats(burn_in=2, thin=3)

        for _ in range(11):
            es.observe(state, accepted=True)

        # Steps 1..11, burn_in=2 skips steps 1,2
        # After burn-in: steps 3..11 = 9 steps
        # thin=3: record at offset 0,3,6 = steps 3,6,9 = 3 samples
        assert es.n_samples == 3

    def test_report_structure(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=True)
        es = EnsembleStats()
        es.observe(state, accepted=True)

        report = es.report()
        assert report["n_samples"] == 1
        assert "boundary_frequencies" in report
        assert "facility_frequencies" in report
        assert "essential_facilities" in report
        assert "substitutable_facilities" in report
        assert "robust_boundaries" in report
        assert "fragile_boundaries" in report
        assert "capacity" in report

    def test_with_no_travel_times(self, small_grid):
        """Ensemble works even without travel times (fake facility)."""
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=False)
        es = EnsembleStats()
        es.observe(state, accepted=True)

        assert es.n_samples == 1
        assert len(es.boundary.frequencies()) > 0
        # Facility frequencies empty because FakeFacility has no centers
        assert len(es.facility.frequencies()) == 0


# ---------------------------------------------------------------------------
# MarkovChain callback integration test
# ---------------------------------------------------------------------------

class TestEnsembleChainIntegration:
    def test_callback_invoked(self, small_grid):
        """EnsembleStats.observe receives calls from MarkovChain callbacks."""
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=False)

        ensemble = EnsembleStats()
        chain = MarkovChain(
            proposal=lambda s: s,  # identity (no-op)
            constraints=lambda p: True,
            accept=always_accept,
            initial_state=state,
            total_steps=6,
            callbacks=[ensemble.observe],
        )

        list(chain)

        # 6 total_steps yields initial + 5 __next__ calls
        # callbacks fire on steps 1..5 (not initial), so 5 observations
        assert ensemble.n_samples == 5

    def test_callback_with_burn_in(self, small_grid):
        set_seed(42)
        state = _make_state(small_grid, with_travel_times=False)

        ensemble = EnsembleStats(burn_in=2)
        chain = MarkovChain(
            proposal=lambda s: s,
            constraints=lambda p: True,
            accept=always_accept,
            initial_state=state,
            total_steps=6,
            callbacks=[ensemble.observe],
        )

        list(chain)
        # 5 callback calls - 2 burn-in = 3 recorded
        assert ensemble.n_samples == 3
