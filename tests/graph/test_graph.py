"""
Tests for falcomchain.graph.graph — Graph, FrozenGraph, and utilities.
"""

import json
import os
import tempfile
import warnings

from falcomchain.random import set_seed

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from falcomchain.graph.graph import (
    FrozenGraph,
    Graph,
    add_boundary_perimeters,
    check_dataframe,
    json_serialize,
)
from falcomchain.graph.grid import Grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_nx_graph():
    """A 3-node path graph with basic attributes."""
    G = nx.path_graph(3)
    for n in G.nodes:
        G.nodes[n]["demand"] = 100
        G.nodes[n]["area"] = 1.0
        G.nodes[n]["candidate"] = 1 if n == 1 else 0
    return G


@pytest.fixture
def grid_graph():
    """A 6x5 grid via the Grid class."""
    set_seed(42)
    return Grid(dimensions=(6, 5), num_candidates=6, density="uniform").graph


@pytest.fixture
def small_geodataframe():
    """A 2x2 grid of square polygons as a GeoDataFrame."""
    polys = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
        box(1, 1, 2, 2),
    ]
    gdf = gpd.GeoDataFrame(
        {"demand": [10, 20, 30, 40], "candidate": [1, 0, 0, 1]},
        geometry=polys,
    )
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestGraphFromNetworkx:
    def test_creates_graph_instance(self, simple_nx_graph):
        g = Graph.from_networkx(simple_nx_graph)
        assert isinstance(g, Graph)
        assert isinstance(g, nx.Graph)

    def test_preserves_nodes_and_edges(self, simple_nx_graph):
        g = Graph.from_networkx(simple_nx_graph)
        assert set(g.nodes) == set(simple_nx_graph.nodes)
        assert set(g.edges) == set(simple_nx_graph.edges)

    def test_preserves_node_attributes(self, simple_nx_graph):
        g = Graph.from_networkx(simple_nx_graph)
        assert g.nodes[1]["demand"] == 100
        assert g.nodes[1]["candidate"] == 1


class TestGraphFromGeoDataFrame:
    def test_creates_graph_from_geodataframe(self, small_geodataframe):
        g = Graph.from_geodataframe(small_geodataframe, adjacency="rook")
        assert isinstance(g, Graph)
        assert g.number_of_nodes() == 4

    def test_rook_adjacency_is_correct(self, small_geodataframe):
        g = Graph.from_geodataframe(small_geodataframe, adjacency="rook")
        # In a 2x2 grid with rook adjacency, each corner has 2 neighbors
        # Total edges: 4 (horizontal/vertical shared boundaries)
        assert g.number_of_edges() == 4

    def test_adds_area_attribute(self, small_geodataframe):
        g = Graph.from_geodataframe(small_geodataframe, adjacency="rook")
        for node in g.nodes:
            assert "area" in g.nodes[node]

    def test_adds_boundary_node_attribute(self, small_geodataframe):
        g = Graph.from_geodataframe(small_geodataframe, adjacency="rook")
        # All 4 polygons are on the boundary of the 2x2 grid
        for node in g.nodes:
            assert g.nodes[node]["boundary_node"] is True

    def test_adds_columns_as_attributes(self, small_geodataframe):
        g = Graph.from_geodataframe(
            small_geodataframe, adjacency="rook", cols_to_add=["demand", "candidate"]
        )
        assert g.nodes[0]["demand"] == 10
        assert g.nodes[3]["candidate"] == 1

    def test_stores_crs(self, small_geodataframe):
        g = Graph.from_geodataframe(small_geodataframe, adjacency="rook")
        assert g.graph["crs"] is not None


# ---------------------------------------------------------------------------
# Graph JSON round-trip
# ---------------------------------------------------------------------------

class TestGraphJson:
    def test_save_and_load_roundtrip(self, simple_nx_graph):
        g = Graph.from_networkx(simple_nx_graph)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            g.to_json(path)
            loaded = Graph.from_json(path)
            assert set(loaded.nodes) == set(g.nodes)
            assert set(loaded.edges) == set(g.edges)
            for n in loaded.nodes:
                assert loaded.nodes[n]["demand"] == g.nodes[n]["demand"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Graph properties and methods
# ---------------------------------------------------------------------------

class TestGraphProperties:
    def test_repr(self, grid_graph):
        r = repr(grid_graph)
        assert "30 nodes" in r
        assert "49 edges" in r

    def test_node_indices(self, grid_graph):
        assert grid_graph.node_indices == set(grid_graph.nodes)

    def test_edge_indices(self, grid_graph):
        assert grid_graph.edge_indices == set(grid_graph.edges)

    def test_lookup(self, grid_graph):
        node = list(grid_graph.nodes)[0]
        assert grid_graph.lookup(node, "demand") == grid_graph.nodes[node]["demand"]

    def test_islands_empty_for_grid(self, grid_graph):
        assert len(grid_graph.islands) == 0

    def test_islands_detected(self):
        g = Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(1, 2)
        assert g.islands == {0}

    def test_warn_for_islands(self):
        g = Graph()
        g.add_node(0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g.warn_for_islands()
            assert len(w) == 1
            assert "islands" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# FrozenGraph
# ---------------------------------------------------------------------------

class TestFrozenGraph:
    def test_creation(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        assert len(fg) == 30

    def test_immutable(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        with pytest.raises(nx.NetworkXError):
            fg.graph.add_node(999)

    def test_neighbors_cached(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        node = list(grid_graph.nodes)[5]
        n1 = fg.neighbors(node)
        n2 = fg.neighbors(node)
        assert n1 is n2  # same object from cache

    def test_node_indices(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        assert set(fg.node_indices) == set(grid_graph.nodes)

    def test_lookup(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        node = list(grid_graph.nodes)[0]
        assert fg.lookup(node, "demand") == grid_graph.nodes[node]["demand"]

    def test_subgraph(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        nodes = list(grid_graph.nodes)[:5]
        sub = fg.subgraph(nodes)
        assert isinstance(sub, FrozenGraph)
        assert len(sub) == 5

    def test_iteration(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        nodes = list(fg)
        assert len(nodes) == 30

    def test_getitem(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        node = list(grid_graph.nodes)[0]
        # fg[node] returns adjacency view for that node
        neighbors = fg[node]
        assert len(neighbors) > 0

    def test_degree(self, grid_graph):
        fg = FrozenGraph(grid_graph)
        corner = (0, 0)
        assert fg.degree(corner) == 2  # corner of grid has 2 neighbors


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class TestGrid:
    def test_uniform_grid(self):
        set_seed(42)
        grid = Grid(dimensions=(4, 4), num_candidates=4, density="uniform")
        g = grid.graph
        assert isinstance(g, Graph)
        assert g.number_of_nodes() == 16
        # All demands should be 50 (uniform default)
        for n in g.nodes:
            assert g.nodes[n]["demand"] == 50

    def test_corners_density(self):
        set_seed(42)
        grid = Grid(dimensions=(6, 6), num_candidates=6, density="corners", threshold=(2, 2))
        g = grid.graph
        # Corner nodes should have demand 70, others 30
        assert g.nodes[(0, 0)]["demand"] == 70
        assert g.nodes[(3, 3)]["demand"] == 30

    def test_candidate_count(self):
        set_seed(42)
        grid = Grid(dimensions=(5, 5), num_candidates=5, density="uniform")
        g = grid.graph
        cand_count = sum(1 for n in g.nodes if g.nodes[n]["candidate"])
        assert cand_count == 5

    def test_coordinates_assigned(self):
        set_seed(42)
        grid = Grid(dimensions=(3, 3), num_candidates=2, density="uniform")
        g = grid.graph
        for n in g.nodes:
            assert "C_X" in g.nodes[n]
            assert "C_Y" in g.nodes[n]

    def test_boundary_nodes_tagged(self):
        set_seed(42)
        grid = Grid(dimensions=(4, 4), num_candidates=4, density="uniform")
        g = grid.graph
        # (0,0) is a corner -> boundary
        assert g.nodes[(0, 0)]["boundary_node"] is True
        # (1,1) is interior -> not boundary
        assert g.nodes[(1, 1)]["boundary_node"] is False

    def test_invalid_dimensions(self):
        with pytest.raises(Exception, match="Dimension must be 2"):
            Grid(dimensions=(3,), num_candidates=1, density="uniform")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestJsonSerialize:
    def test_int64(self):
        val = pd.array([42], dtype="Int64")[0]
        assert json_serialize(val) == 42

    def test_non_int_returns_none(self):
        assert json_serialize("hello") is None


class TestCheckDataframe:
    def test_warns_on_na(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_dataframe(df)
            assert any("NA" in str(x.message) for x in w)

    def test_no_warning_when_clean(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_dataframe(df)
            assert len(w) == 0
