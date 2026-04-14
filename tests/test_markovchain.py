"""
Tests for:
  - falcomchain/markovchain/accept.py   (always_accept, metropolis_hastings)
  - falcomchain/markovchain/state.py    (ChainState)
  - falcomchain/markovchain/energy.py   (compute_energy, compute_energy_delta)
  - falcomchain/partition/flows.py      (Flow)
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from falcomchain.markovchain.accept import always_accept, metropolis_hastings
from falcomchain.markovchain.energy import compute_energy, compute_energy_delta
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.flows import Flow


# ---------------------------------------------------------------------------
# Helpers — minimal stubs that satisfy the interfaces under test
# ---------------------------------------------------------------------------

def _make_partition(parts=None, mapping=None, candidates=None, teams=None,
                    graph_nodes=None, flow=None):
    """Return a minimal Partition-like namespace."""
    parts = parts or {1: frozenset([1, 2]), 2: frozenset([3, 4])}
    mapping = mapping or {1: 1, 2: 1, 3: 2, 4: 2}
    candidates = candidates or {1: frozenset([1]), 2: frozenset([3])}
    teams = teams or {1: 1, 2: 1}

    # graph with demand attribute
    g = SimpleNamespace()
    g.nodes = graph_nodes or {
        1: {"demand": 10, "candidate": 1},
        2: {"demand": 10, "candidate": 0},
        3: {"demand": 10, "candidate": 1},
        4: {"demand": 10, "candidate": 0},
    }

    assignment = SimpleNamespace(
        parts=parts,
        mapping=mapping,
        candidates=candidates,
        teams=teams,
        travel_times=None,   # set per-test when needed
    )

    p = SimpleNamespace(
        assignment=assignment,
        graph=g,
        parts=parts,
        teams=teams,
        candidates=candidates,
        capacity_level=1,
        step=1,
        flow=flow or Flow.initial(_make_flip(parts)),
        parent=None,
    )
    return p


def _make_flip(parts):
    """Minimal Flip-like namespace with new_ids."""
    return SimpleNamespace(new_ids=frozenset(parts.keys()))


def _make_facility(centers=None, radii=None):
    """Minimal FacilityAssignment stub."""
    centers = centers or {1: 1, 2: 3}
    radii = radii or {1: 5.0, 2: 5.0}
    return SimpleNamespace(centers=centers, radii=radii)


def _make_state(energy=0.0, beta=1.0, log_proposal_ratio=0.0,
                feasible=True, partition=None, facility=None):
    """Build a ChainState via __new__ to bypass normal construction."""
    state = ChainState.__new__(ChainState)
    state.partition = partition or _make_partition()
    state.facility = facility or _make_facility()
    state.super_facility = None
    state.energy = energy
    state.log_proposal_ratio = log_proposal_ratio
    state.beta = beta
    state.feasible = feasible
    state.energy_fn = None
    state._recorder = None
    return state


# ---------------------------------------------------------------------------
# Flow tests
# ---------------------------------------------------------------------------

class TestFlow:
    def test_initial_has_no_node_or_candidate_flows(self):
        flip = _make_flip({1: frozenset([1, 2]), 2: frozenset([3])})
        flow = Flow.initial(flip)
        assert flow.node_flows is None
        assert flow.candidate_flows is None

    def test_initial_part_flows_in_equals_new_ids(self):
        flip = _make_flip({1: frozenset([1, 2]), 2: frozenset([3])})
        flow = Flow.initial(flip)
        assert flow.part_flows["in"] == frozenset({1, 2})
        assert flow.part_flows["out"] == set()

    def test_flow_fields_accessible(self):
        node_flows = {1: {"in": {2}, "out": {1}}}
        part_flows = {"in": {3}, "out": {1}}
        candidate_flows = {1: {"in": set(), "out": set()}}
        flow = Flow(node_flows=node_flows, part_flows=part_flows,
                    candidate_flows=candidate_flows)
        assert flow.node_flows is node_flows
        assert flow.part_flows is part_flows
        assert flow.candidate_flows is candidate_flows


# ---------------------------------------------------------------------------
# accept.py tests
# ---------------------------------------------------------------------------

class TestAlwaysAccept:
    def test_returns_true(self):
        state = _make_state()
        assert always_accept(state, state) is True

    def test_ignores_energy(self):
        proposed = _make_state(energy=1e9)
        current = _make_state(energy=0.0)
        assert always_accept(proposed, current) is True


class TestMetropolisHastings:
    def test_rejects_infeasible(self):
        proposed = _make_state(feasible=False)
        current = _make_state()
        assert metropolis_hastings(proposed, current) is False

    def test_accepts_when_beta_zero(self):
        proposed = _make_state(energy=1e6, beta=0.0)
        current = _make_state(energy=0.0, beta=0.0)
        assert metropolis_hastings(proposed, current) is True

    def test_always_accepts_lower_energy(self):
        # lower energy → positive log_alpha → always accepted
        proposed = _make_state(energy=0.0, beta=1.0)
        current = _make_state(energy=100.0, beta=1.0)
        # log_alpha = -1*(0-100) + 0 = 100 >> 0, so log(random()) <= 100 always
        accepted = metropolis_hastings(proposed, current)
        assert accepted is True

    def test_always_rejects_infinite_energy_increase(self):
        proposed = _make_state(energy=1e300, beta=1.0)
        current = _make_state(energy=0.0, beta=1.0)
        # log_alpha = -1e300, math.log(random()) is at most ~0 > -1e300 is false
        with patch("falcomchain.markovchain.accept.rng.random", return_value=0.5):
            result = metropolis_hastings(proposed, current)
        assert result is False

    def test_log_proposal_ratio_shifts_acceptance(self):
        # Equal energy but large positive log_proposal_ratio → accepts
        proposed = _make_state(energy=10.0, beta=1.0, log_proposal_ratio=100.0)
        current = _make_state(energy=10.0, beta=1.0)
        assert metropolis_hastings(proposed, current) is True

    def test_log_proposal_ratio_negative_inhibits_acceptance(self):
        # Equal energy but large negative log_proposal_ratio → rejects
        proposed = _make_state(energy=10.0, beta=1.0, log_proposal_ratio=-1e300)
        current = _make_state(energy=10.0, beta=1.0)
        with patch("falcomchain.markovchain.accept.rng.random", return_value=0.5):
            result = metropolis_hastings(proposed, current)
        assert result is False


# ---------------------------------------------------------------------------
# ChainState tests
# ---------------------------------------------------------------------------

class TestChainState:
    def test_repr_contains_step_energy_beta_feasible(self):
        state = _make_state(energy=3.14, beta=0.5)
        r = repr(state)
        assert "energy=3.1400" in r
        assert "beta=0.5" in r
        assert "feasible=True" in r

    def test_pass_through_properties(self):
        partition = _make_partition()
        state = _make_state(partition=partition)
        assert state.parts is partition.parts
        assert state.assignment is partition.assignment
        assert state.graph is partition.graph
        assert state.teams is partition.teams
        assert state.candidates is partition.candidates
        assert state.capacity_level == 1
        assert state.step == 1

    def test_centers_come_from_facility(self):
        facility = _make_facility(centers={1: 99, 2: 88})
        state = _make_state(facility=facility)
        assert state.centers == {1: 99, 2: 88}

    def test_radii_come_from_facility(self):
        facility = _make_facility(radii={1: 7.0, 2: 3.5})
        state = _make_state(facility=facility)
        assert state.radii == {1: 7.0, 2: 3.5}

    def test_next_carries_beta_forward(self):
        current = _make_state(beta=2.5)
        proposed_partition = _make_partition()
        # patch FacilityAssignment.updated so we don't need real travel_times
        with patch("falcomchain.markovchain.state.FacilityAssignment.updated",
                   return_value=_make_facility()):
            proposed = current.next(
                partition=proposed_partition,
                energy=5.0,
                log_proposal_ratio=0.1,
                feasible=True,
            )
        assert proposed.beta == 2.5
        assert proposed.energy == 5.0
        assert proposed.log_proposal_ratio == 0.1
        assert proposed.feasible is True

    def test_next_proposed_state_is_new_object(self):
        current = _make_state()
        with patch("falcomchain.markovchain.state.FacilityAssignment.updated",
                   return_value=_make_facility()):
            proposed = current.next(
                partition=_make_partition(),
                energy=1.0,
                log_proposal_ratio=0.0,
                feasible=True,
            )
        assert proposed is not current

    def test_len_delegates_to_partition(self):
        parts = {1: frozenset([1]), 2: frozenset([2]), 3: frozenset([3])}

        class FakePartition:
            assignment = SimpleNamespace(parts=parts, mapping={1:1,2:2,3:3},
                                         candidates={}, teams={})
            graph = SimpleNamespace(nodes={n: {"demand": 10, "candidate": 0}
                                           for n in [1, 2, 3]})
            capacity_level = 1
            step = 1
            flow = Flow.initial(SimpleNamespace(new_ids=frozenset(parts)))
            parent = None
            def __len__(self):
                return sum(len(v) for v in parts.values())

        state = _make_state(partition=FakePartition())
        assert len(state) == 3


# ---------------------------------------------------------------------------
# energy.py tests
# ---------------------------------------------------------------------------

def _travel_times_for(parts, factor=1.0):
    """Build a symmetric travel_times dict: (center, node) → factor * 1."""
    tt = {}
    for district, nodes in parts.items():
        center = next(iter(nodes))  # first node as center
        for node in nodes:
            tt[(center, node)] = factor * 1.0
    return tt


class TestComputeEnergy:
    def test_zero_when_all_travel_times_zero(self):
        parts = {1: frozenset([1, 2]), 2: frozenset([3, 4])}
        centers = {1: 1, 2: 3}
        tt = {(1, 1): 0.0, (1, 2): 0.0, (3, 3): 0.0, (3, 4): 0.0}
        assignment = SimpleNamespace(
            parts=parts,
            travel_times=tt,
        )
        graph_nodes = {n: {"demand": 10} for n in [1, 2, 3, 4]}
        graph = SimpleNamespace(nodes=graph_nodes)
        facility = SimpleNamespace(centers=centers)
        state = SimpleNamespace(assignment=assignment, graph=graph, facility=facility)

        assert compute_energy(state) == 0.0

    def test_energy_is_sum_of_demand_times_travel_time(self):
        parts = {1: frozenset([1, 2])}
        centers = {1: 1}
        tt = {(1, 1): 2.0, (1, 2): 4.0}
        assignment = SimpleNamespace(parts=parts, travel_times=tt)
        graph = SimpleNamespace(nodes={1: {"demand": 5}, 2: {"demand": 3}})
        facility = SimpleNamespace(centers=centers)
        state = SimpleNamespace(assignment=assignment, graph=graph, facility=facility)

        # E = 5*2 + 3*4 = 10 + 12 = 22
        assert compute_energy(state) == pytest.approx(22.0)

    def test_skips_district_with_no_center(self):
        parts = {1: frozenset([1, 2]), 2: frozenset([3])}
        centers = {1: 1, 2: None}   # district 2 has no center yet
        tt = {(1, 1): 1.0, (1, 2): 1.0}
        assignment = SimpleNamespace(parts=parts, travel_times=tt)
        graph = SimpleNamespace(nodes={n: {"demand": 10} for n in [1, 2, 3]})
        facility = SimpleNamespace(centers=centers)
        state = SimpleNamespace(assignment=assignment, graph=graph, facility=facility)

        assert compute_energy(state) == pytest.approx(20.0)

    def test_energy_scales_with_demand(self):
        parts = {1: frozenset([1])}
        centers = {1: 1}
        tt = {(1, 1): 3.0}
        assignment = SimpleNamespace(parts=parts, travel_times=tt)
        graph = SimpleNamespace(nodes={1: {"demand": 7}})
        facility = SimpleNamespace(centers=centers)
        state = SimpleNamespace(assignment=assignment, graph=graph, facility=facility)
        assert compute_energy(state) == pytest.approx(21.0)


class TestComputeEnergyDelta:
    def _build_state(self, parts, centers, travel_times, demands,
                     node_flows=None, part_flows=None):
        assignment = SimpleNamespace(parts=parts, travel_times=travel_times)
        graph = SimpleNamespace(nodes={n: {"demand": d} for n, d in demands.items()})
        facility = SimpleNamespace(centers=centers)
        flow = SimpleNamespace(
            node_flows=node_flows or {},
            part_flows=part_flows or {"in": set(), "out": set()},
        )
        partition = SimpleNamespace(flow=flow)
        return SimpleNamespace(
            assignment=assignment, graph=graph, facility=facility,
            partition=partition,
        )

    def test_delta_zero_when_nothing_changed(self):
        parts = {1: frozenset([1, 2])}
        centers = {1: 1}
        tt = {(1, 1): 2.0, (1, 2): 3.0}
        demands = {1: 10, 2: 10}
        # no node flows → changed_parts is empty
        proposed = self._build_state(parts, centers, tt, demands,
                                     node_flows={}, part_flows={"in": set(), "out": set()})
        current = self._build_state(parts, centers, tt, demands)
        assert compute_energy_delta(proposed, current) == pytest.approx(0.0)

    def test_delta_reflects_changed_district(self):
        # current: district 1 has nodes {1,2}, center=1, tt={1:0, 2:5}
        # proposed: node 2 moves to district 2; district 1 now only {1}
        old_parts = {1: frozenset([1, 2]), 2: frozenset([3])}
        new_parts = {1: frozenset([1]),    2: frozenset([2, 3])}
        tt = {(1, 1): 0.0, (1, 2): 5.0, (3, 2): 1.0, (3, 3): 0.0}
        demands = {1: 10, 2: 10, 3: 10}

        old_centers = {1: 1, 2: 3}
        new_centers = {1: 1, 2: 3}

        # node 2 moved from district 1 → district 2
        node_flows = {1: {"in": set(), "out": {2}},
                      2: {"in": {2},   "out": set()}}

        proposed = self._build_state(new_parts, new_centers, tt, demands,
                                     node_flows=node_flows,
                                     part_flows={"in": set(), "out": set()})
        current = self._build_state(old_parts, old_centers, tt, demands)

        delta = compute_energy_delta(proposed, current)
        full_new = compute_energy(proposed)
        full_old = compute_energy(current)
        assert delta == pytest.approx(full_new - full_old, rel=1e-9)
