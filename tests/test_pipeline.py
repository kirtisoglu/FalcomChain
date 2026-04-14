"""
End-to-end tests for the FalcomChain pipeline.

Uses a small synthetic grid (6x5 = 30 nodes) to test the full path:
  spanning tree -> bipartition -> recursive partitioning -> Partition ->
  MarkovChain iteration -> Recorder output.
"""

import json
import math
import os
import shutil
import tempfile
from functools import partial
from unittest.mock import patch

import networkx as nx
import pytest

from falcomchain.random import rng, set_seed

from falcomchain.graph import Graph
from falcomchain.graph.grid import Grid
from falcomchain.tree.tree import (
    Cut,
    CutParams,
    Flip,
    SpanningTree,
    _part_nodes,
    accumulate_tree,
    bipartition_tree,
    capacitated_recursive_tree,
    find_edge_cuts,
    find_superedge_cuts,
    one_sided_cut,
    random_spanning_tree,
    two_sided_cut,
    uniform_spanning_tree,
)
from falcomchain.tree.snapshot import Recorder
from falcomchain.partition import Partition
from falcomchain.constraints import Validator
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.accept import always_accept, metropolis_hastings
from falcomchain.markovchain.energy import compute_energy, compute_energy_delta
from falcomchain.markovchain.facility import FacilityAssignment, SuperFacilityAssignment
from falcomchain.markovchain.chain import MarkovChain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """6x5 grid with 30 nodes, demand=100 each, ~6 candidates."""
    set_seed(42)
    grid = Grid(
        dimensions=(6, 5),
        num_candidates=6,
        density="uniform",
    )
    return grid.graph


@pytest.fixture
def small_nx_graph():
    """A plain networkx grid for tree-level tests (no Graph wrapper needed).
    6x5 = 30 nodes, 6 candidates, demand=100 each -> total 3000."""
    set_seed(42)
    G = nx.generators.lattice.grid_2d_graph(6, 5)
    candidates = rng.sample(list(G.nodes), 6)
    for node in G.nodes:
        G.nodes[node]["demand"] = 100
        G.nodes[node]["area"] = 1
        G.nodes[node]["candidate"] = 1 if node in candidates else 0
        G.nodes[node]["C_X"] = float(node[0])
        G.nodes[node]["C_Y"] = float(node[1])
    return G


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for Recorder output, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="falcom_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Spanning tree tests
# ---------------------------------------------------------------------------

class TestRandomSpanningTree:
    def test_returns_connected_tree(self, small_nx_graph):
        tree = random_spanning_tree(small_nx_graph)
        assert nx.is_tree(tree)
        assert set(tree.nodes) == set(small_nx_graph.nodes)

    def test_preserves_node_attributes(self, small_nx_graph):
        tree = random_spanning_tree(small_nx_graph)
        for node in tree.nodes:
            assert "demand" in tree.nodes[node]
            assert "candidate" in tree.nodes[node]

    def test_has_n_minus_1_edges(self, small_nx_graph):
        tree = random_spanning_tree(small_nx_graph)
        assert tree.number_of_edges() == small_nx_graph.number_of_nodes() - 1


class TestUniformSpanningTree:
    def test_returns_connected_tree(self, small_nx_graph):
        tree = uniform_spanning_tree(small_nx_graph)
        assert nx.is_tree(tree)
        assert set(tree.nodes) == set(small_nx_graph.nodes)

    def test_preserves_node_attributes(self, small_nx_graph):
        tree = uniform_spanning_tree(small_nx_graph)
        for node in tree.nodes:
            assert "demand" in tree.nodes[node]

    def test_has_n_minus_1_edges(self, small_nx_graph):
        tree = uniform_spanning_tree(small_nx_graph)
        assert tree.number_of_edges() == small_nx_graph.number_of_nodes() - 1

    def test_different_seeds_produce_different_trees(self, small_nx_graph):
        set_seed(1)
        t1 = uniform_spanning_tree(small_nx_graph)
        set_seed(2)
        t2 = uniform_spanning_tree(small_nx_graph)
        # Very unlikely to be identical with different seeds
        assert set(t1.edges) != set(t2.edges)


# ---------------------------------------------------------------------------
# SpanningTree accumulation tests
# ---------------------------------------------------------------------------

class TestSpanningTreeAccumulation:
    def test_root_demand_equals_total(self, small_nx_graph):
        set_seed(42)
        tree = random_spanning_tree(small_nx_graph)
        params = CutParams(
            ideal_demand=500, epsilon=0.2, capacity_level=2, n_teams=6, two_sided=False
        )
        h = SpanningTree(graph=tree, params=params)
        assert h.total_demand == 3000  # 30 nodes * 100 demand

    def test_root_candidates_equals_total(self, small_nx_graph):
        set_seed(42)
        tree = random_spanning_tree(small_nx_graph)
        params = CutParams(
            ideal_demand=500, epsilon=0.2, capacity_level=2, n_teams=6, two_sided=False
        )
        h = SpanningTree(graph=tree, params=params)
        # After accumulation, root's candidate count = total candidates
        assert h.graph.nodes[h.root]["candidate"] == len(h.candidate_nodes)

    def test_candidate_nodes_stored_before_accumulation(self, small_nx_graph):
        set_seed(42)
        tree = random_spanning_tree(small_nx_graph)
        params = CutParams(
            ideal_demand=500, epsilon=0.2, capacity_level=2, n_teams=6, two_sided=False
        )
        h = SpanningTree(graph=tree, params=params)
        # candidate_nodes should contain exactly the original candidate nodes
        assert len(h.candidate_nodes) == 6
        for cn in h.candidate_nodes:
            # Original value was 1, but after accumulation it's >= 1
            assert cn in small_nx_graph.nodes


# ---------------------------------------------------------------------------
# bipartition_tree tests
# ---------------------------------------------------------------------------

class TestBipartitionTree:
    def test_finds_balanced_cut(self, small_nx_graph):
        set_seed(42)
        # 100 nodes, 10000 total demand, 10 teams, target 1000/team
        cut, log_ratio = bipartition_tree(
            graph=small_nx_graph,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
            n_teams=6,
            two_sided=False,
            supergraph=False,
        )
        assert isinstance(cut, Cut)
        assert cut.assigned_teams >= 1
        assert cut.demand > 0
        assert cut.psi > 0
        assert isinstance(log_ratio, float)

    def test_cut_subnodes_are_subset(self, small_nx_graph):
        set_seed(42)
        cut, _ = bipartition_tree(
            graph=small_nx_graph,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
            n_teams=6,
            two_sided=False,
            supergraph=False,
        )
        assert cut.subnodes.issubset(set(small_nx_graph.nodes))
        assert len(cut.subnodes) > 0
        assert len(cut.subnodes) < small_nx_graph.number_of_nodes()

    def test_two_sided_cut(self, small_nx_graph):
        set_seed(42)
        cut, log_ratio = bipartition_tree(
            graph=small_nx_graph,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
            n_teams=6,
            two_sided=True,
            supergraph=False,
        )
        assert isinstance(cut, Cut)
        complement_demand = 3000 - cut.demand
        assert complement_demand > 0

    def test_respects_tree_sampler_parameter(self, small_nx_graph):
        """Verify that custom tree_sampler is called."""
        call_count = [0]
        def counting_sampler(graph):
            call_count[0] += 1
            return random_spanning_tree(graph)

        set_seed(42)
        bipartition_tree(
            graph=small_nx_graph,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
            n_teams=6,
            two_sided=False,
            supergraph=False,
            tree_sampler=counting_sampler,
        )
        assert call_count[0] >= 1

    def test_log_ratio_is_negative(self, small_nx_graph):
        """log(psi_chosen / total_psi) should be <= 0."""
        set_seed(42)
        _, log_ratio = bipartition_tree(
            graph=small_nx_graph,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
            n_teams=6,
            two_sided=False,
            supergraph=False,
        )
        assert log_ratio <= 0.0


# ---------------------------------------------------------------------------
# capacitated_recursive_tree tests
# ---------------------------------------------------------------------------

class TestCapacitatedRecursiveTree:
    def test_produces_valid_flip(self, small_nx_graph):
        set_seed(42)
        flip = capacitated_recursive_tree(
            graph=small_nx_graph,
            n_teams=6,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
        )
        assert isinstance(flip, Flip)
        # Every node should be assigned
        assert set(flip.flips.keys()) == set(small_nx_graph.nodes)
        # Teams should sum to n_teams
        assert sum(flip.team_flips.values()) == 6

    def test_each_district_has_at_least_one_team(self, small_nx_graph):
        set_seed(42)
        flip = capacitated_recursive_tree(
            graph=small_nx_graph,
            n_teams=6,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
        )
        for teams in flip.team_flips.values():
            assert teams >= 1

    def test_each_district_respects_capacity(self, small_nx_graph):
        set_seed(42)
        flip = capacitated_recursive_tree(
            graph=small_nx_graph,
            n_teams=6,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
        )
        for teams in flip.team_flips.values():
            assert teams <= 3

    def test_districts_are_contiguous(self, small_nx_graph):
        set_seed(42)
        flip = capacitated_recursive_tree(
            graph=small_nx_graph,
            n_teams=6,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
        )
        # Group nodes by district
        districts = {}
        for node, dist_id in flip.flips.items():
            districts.setdefault(dist_id, set()).add(node)

        for dist_id, nodes in districts.items():
            subgraph = small_nx_graph.subgraph(nodes)
            assert nx.is_connected(subgraph), f"District {dist_id} is not contiguous"

    def test_accumulates_log_proposal_ratio(self, small_nx_graph):
        set_seed(42)
        flip = capacitated_recursive_tree(
            graph=small_nx_graph,
            n_teams=6,
            demand_target=500,
            epsilon=0.3,
            capacity_level=3,
        )
        assert isinstance(flip.log_proposal_ratio, float)
        assert flip.log_proposal_ratio <= 0.0


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------

class TestPartition:
    def test_from_random_assignment(self, small_grid):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid,
            epsilon=0.3,
            demand_target=500,
            assignment_class=None,
            capacity_level=3,
        )
        assert len(partition) > 0
        # All nodes accounted for
        total_nodes = sum(len(nodes) for nodes in partition.parts.values())
        assert total_nodes == 30

    def test_parts_and_teams_consistent(self, small_grid):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid,
            epsilon=0.3,
            demand_target=500,
            assignment_class=None,
            capacity_level=3,
        )
        # Every district in parts should have a teams entry
        for part_id in partition.parts:
            assert part_id in partition.teams

    def test_supergraph_exists(self, small_grid):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid,
            epsilon=0.3,
            demand_target=500,
            assignment_class=None,
            capacity_level=3,
        )
        assert partition.supergraph is not None
        assert isinstance(partition.supergraph, nx.Graph)

    def test_each_district_has_candidate(self, small_grid):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid,
            epsilon=0.3,
            demand_target=500,
            assignment_class=None,
            capacity_level=3,
        )
        for part_id, cands in partition.candidates.items():
            assert len(cands) > 0, f"District {part_id} has no candidates"


# ---------------------------------------------------------------------------
# FacilityAssignment tests
# ---------------------------------------------------------------------------

class TestFacilityAssignment:
    def _make_state_with_travel_times(self, small_grid):
        """Helper: build a ChainState with fake travel times."""
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid,
            epsilon=0.3,
            demand_target=500,
            assignment_class=None,
            capacity_level=3,
        )
        # Set up fake travel times: distance = |x1-x2| + |y1-y2|
        from falcomchain.partition.assignment import Assignment
        travel_times = {}
        g = partition.graph.graph
        for n1 in g.nodes:
            for n2 in g.nodes:
                d = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])
                travel_times[(n1, n2)] = float(d)
        Assignment.travel_times = travel_times

        state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
        return state

    def test_centers_computed_for_all_districts(self, small_grid):
        state = self._make_state_with_travel_times(small_grid)
        assert len(state.facility.centers) == len(state.parts)

    def test_radii_are_finite(self, small_grid):
        state = self._make_state_with_travel_times(small_grid)
        for r in state.facility.radii.values():
            assert r < float("inf")

    def test_center_is_a_candidate(self, small_grid):
        state = self._make_state_with_travel_times(small_grid)
        for part_id, center in state.facility.centers.items():
            assert center in state.candidates[part_id]


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def test_all_pass(self):
        v = Validator([lambda p: True, lambda p: True])
        assert v(None) is True

    def test_one_fails(self):
        v = Validator([lambda p: True, lambda p: False])
        assert v(None) is False

    def test_empty_constraints(self):
        v = Validator([])
        assert v(None) is True

    def test_raises_on_non_bool(self):
        v = Validator([lambda p: "yes"])
        with pytest.raises(TypeError):
            v(None)


# ---------------------------------------------------------------------------
# Recorder tests
# ---------------------------------------------------------------------------

class TestRecorder:
    def test_write_header_creates_file(self, small_grid, tmp_output_dir):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid, epsilon=0.3, demand_target=500,
            assignment_class=None, capacity_level=3,
        )
        recorder = Recorder(tmp_output_dir)
        recorder.write_header(small_grid, partition, {"epsilon": 0.1})
        recorder.close()

        fcrec_path = os.path.join(tmp_output_dir, "chain.fcrec")
        assert os.path.exists(fcrec_path)
        assert os.path.getsize(fcrec_path) > 0

    def test_record_step_and_export(self, small_grid, tmp_output_dir):
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid, epsilon=0.3, demand_target=500,
            assignment_class=None, capacity_level=3,
        )
        from falcomchain.partition.assignment import Assignment
        Assignment.travel_times = None

        state = ChainState.__new__(ChainState)
        state.partition = partition
        state.energy = 42.0
        state.beta = 1.0
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

        recorder = Recorder(tmp_output_dir)
        recorder.write_header(small_grid, partition, {"epsilon": 0.3})
        recorder.record_step(state, accepted=True)
        recorder.record_step(state, accepted=False)
        recorder.close()

        # Export and verify
        json_dir = os.path.join(tmp_output_dir, "json")
        Recorder.export_to_json(tmp_output_dir, json_dir)

        manifest_path = os.path.join(json_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["total_steps"] == 2
        assert manifest["graph_nodes"] == 30
        assert "node_candidates" in manifest

        step1_path = os.path.join(json_dir, "step_0001.json")
        assert os.path.exists(step1_path)
        with open(step1_path) as f:
            frame = json.load(f)
        assert frame["step"] == 1
        assert frame["accepted"] is True
        assert "assignment" in frame

    def test_close_updates_step_count(self, small_grid, tmp_output_dir):
        """Verify that close() writes the correct step count into the header."""
        import struct
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid, epsilon=0.3, demand_target=500,
            assignment_class=None, capacity_level=3,
        )
        recorder = Recorder(tmp_output_dir)
        recorder.write_header(small_grid, partition, {})
        recorder._step_count = 42
        recorder.close()

        # Read the header directly and verify n_steps field
        with open(os.path.join(tmp_output_dir, "chain.fcrec"), "rb") as f:
            magic = f.read(5)
            version, n_nodes, n_steps = struct.unpack("<HII", f.read(10))
        assert n_steps == 42


# ---------------------------------------------------------------------------
# MarkovChain integration test
# ---------------------------------------------------------------------------

class TestMarkovChainIntegration:
    def test_chain_iterates_without_error(self, small_grid):
        """Run 3 steps of a chain on the small grid with always_accept."""
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid, epsilon=0.3, demand_target=500,
            assignment_class=None, capacity_level=3,
        )
        from falcomchain.partition.assignment import Assignment
        Assignment.travel_times = None

        state = ChainState.__new__(ChainState)
        state.partition = partition
        state.energy = 0.0
        state.beta = 0.0
        state.log_proposal_ratio = 0.0
        state.feasible = True
        state.energy_fn = None

        class FakeFacility:
            centers = {}
            radii = {}
            def center(self, p): return None
            def radius(self, p): return float("inf")
        state.facility = FakeFacility()
        state.super_facility = None

        # Use always_accept and a no-op constraint
        chain = MarkovChain(
            proposal=lambda s: s,  # identity proposal (no-op)
            constraints=lambda p: True,
            accept=always_accept,
            initial_state=state,
            total_steps=4,
        )

        steps = list(chain)
        assert len(steps) == 4  # initial + 3 steps

    def test_chain_with_recorder(self, small_grid, tmp_output_dir):
        """Verify that Recorder records steps during chain iteration."""
        set_seed(42)
        partition = Partition.from_random_assignment(
            graph=small_grid, epsilon=0.3, demand_target=500,
            assignment_class=None, capacity_level=3,
        )
        from falcomchain.partition.assignment import Assignment
        Assignment.travel_times = None

        state = ChainState.__new__(ChainState)
        state.partition = partition
        state.energy = 0.0
        state.beta = 0.0
        state.log_proposal_ratio = 0.0
        state.feasible = True
        state.energy_fn = None
        state._recorder = None

        class FakeFacility:
            centers = {}
            radii = {}
            def center(self, p): return None
            def radius(self, p): return float("inf")
        state.facility = FakeFacility()
        state.super_facility = None

        recorder = Recorder(tmp_output_dir)
        recorder.write_header(small_grid, partition, {"epsilon": 0.3})

        chain = MarkovChain(
            proposal=lambda s: s,
            constraints=lambda p: True,
            accept=always_accept,
            initial_state=state,
            total_steps=4,
            recorder=recorder,
        )
        list(chain)

        # Binary file should exist and recorder should have counted 3 steps
        assert os.path.exists(os.path.join(tmp_output_dir, "chain.fcrec"))
        assert recorder._step_count == 3


# ---------------------------------------------------------------------------
# Psi score tests
# ---------------------------------------------------------------------------

class TestPsiScore:
    def test_psi_zero_when_no_candidates(self):
        """Node with 0 accumulated candidates should have psi = 0."""
        G = nx.path_graph(3)
        for n in G.nodes:
            G.nodes[n]["demand"] = 10
            G.nodes[n]["area"] = 1
            G.nodes[n]["candidate"] = 0
        params = CutParams(
            ideal_demand=10, epsilon=0.5, capacity_level=1, n_teams=3,
        )
        h = SpanningTree(graph=G, params=params)
        # All nodes have 0 candidates
        for n in G.nodes:
            assert h.psi(n) == 0.0

    def test_psi_equals_phi_when_gamma_zero(self):
        """When gamma=0, psi should equal the candidate count (phi)."""
        G = nx.path_graph(5)
        for n in G.nodes:
            G.nodes[n]["demand"] = 10
            G.nodes[n]["area"] = 1
            G.nodes[n]["candidate"] = 1 if n in (0, 2, 4) else 0
        params = CutParams(
            ideal_demand=10, epsilon=0.5, capacity_level=1, n_teams=5, gamma=0.0,
        )
        h = SpanningTree(graph=G, params=params)
        # Root's accumulated candidate count = total candidates = 3
        assert h.psi(h.root) == float(h.graph.nodes[h.root]["candidate"])

    def test_psi_with_custom_fn(self):
        """Custom psi_fn should be called instead of default."""
        G = nx.path_graph(3)
        for n in G.nodes:
            G.nodes[n]["demand"] = 10
            G.nodes[n]["area"] = 1
            G.nodes[n]["candidate"] = 1

        def custom_psi(phi, gamma, radius):
            return 999.0

        params = CutParams(
            ideal_demand=10, epsilon=0.5, capacity_level=1, n_teams=3,
            gamma=1.0, psi_fn=custom_psi,
        )
        h = SpanningTree(graph=G, params=params)
        assert h.psi(h.root) == 999.0
