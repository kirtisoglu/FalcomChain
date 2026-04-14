from typing import Optional
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import box

from falcomchain.graph import Graph
from falcomchain.partition import Partition
from falcomchain.random import set_seed
from falcomchain.tree.tree import CutParams, SpanningTree

set_seed(2025)


@pytest.fixture
def tree_with_twenty_nodes():
    """Returns a tree that looks like this:

    1 - 2 - 3 - 4
        |       |
    5 - 6 - 7   8 - 9 - 10
    |       |       |
    11-12   13      14
    |
    15-16-17
    |
    18-19
    |
    20

    """
    graph = Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (2, 6),
            (4, 8),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (5, 11),
            (7, 13),
            (9, 14),
            (11, 12),
            (11, 15),
            (15, 16),
            (16, 17),
            (15, 18),
            (18, 19),
            (18, 20),
        ]
    )
    return graph


@pytest.fixture
def tree_with_attributes(tree_with_twenty_nodes):

    candidates = {
        3,
        7,
        8,
        10,
        12,
        17,
        18,
    }  # for pop target = 40 and n_teams = 5, where total population = 200.
    for node in tree_with_twenty_nodes:
        tree_with_twenty_nodes.nodes[node]["demand"] = 10
        tree_with_twenty_nodes.nodes[node]["area"] = 10
        tree_with_twenty_nodes.nodes[node]["density"] = 1
        if node in candidates:
            tree_with_twenty_nodes.nodes[node]["candidate"] = 1
        else:
            tree_with_twenty_nodes.nodes[node]["candidate"] = 0
    return tree_with_twenty_nodes


@pytest.fixture
def spanningtree_with_forced_root(tree_with_attributes):
    n_teams = 5
    epsilon = 0.3  # population of 30 will be excepted since pop_target = 40 and (1-0.3)*40 < 30
    pop_target = 40
    capacity_level = 1
    column_names = ["population", "area", "candidate", "density"]
    two_sided = True
    supergraph = False

    with patch("falcomchain.random.rng.choice", return_value=2):
        tree = SpanningTree(
            graph=tree_with_attributes,
            params=CutParams(
                ideal_demand=pop_target,
                epsilon=epsilon,
                n_teams=n_teams,
                capacity_level=capacity_level,
                two_sided=two_sided,
            ),
            supergraph=supergraph,
        )
    return tree


@pytest.fixture
def planar_graph():
    """Returns a tree that looks like this:

    1 - 2 - 3 - 4
        |   |   |
    5 - 6 - 7   8 - 9 - 10
    |   |   |       |
    11-12   13  -   14
    |  |
    15-16-17
    |     |
    18-19-21
    |
    20

    """

    new_edges = (19, 21), (17, 21), (17, 21), (13, 14), (3, 7), (12, 16), (6, 12)
    new_nodes = {21}


@pytest.fixture
def geodata_without_candidates():
    """Creates a dummy GeoDataFrame simulating census blocks with geometric and demographic attributes.

    Returns:
        GeoDataFrame: A 3x3 grid of square polygons with synthetic IDs, coordinates, population, and candidate flag.
    """

    # Columns to generate:
    # 'id'           : Integer ID (unique for each row)
    # 'GEOID20'      : Synthetic census block GEOID (string)
    # 'INTPTLAT20'   : Latitude of internal point (float)
    # 'INTPTLON20'   : Longitude of internal point (float)
    # 'geometry'     : Polygon geometry of the block (shapely object)
    # 'centroid'     : Centroid of the polygon (shapely Point)
    # 'population'   : Random population number (int)

    rows = 3
    cols = 3
    cell_size = 1.0

    data = []
    id_counter = 0
    for i in range(rows):
        for j in range(cols):
            x_min = j * cell_size
            y_min = i * cell_size
            geom = box(x_min, y_min, x_min + cell_size, y_min + cell_size)
            centroid = geom.centroid
            data.append(
                {
                    "id": id_counter,
                    "GEOID20": f"170310{i}{j:02}",  # Example GEOID for Chicago blocks
                    "INTPTLAT20": centroid.y,
                    "INTPTLON20": centroid.x,
                    "geometry": geom,
                    "centroid": centroid,
                    "population": 100
                    + 10 * i
                    + j,  # Just a varying value for population
                    "candidate": (i + j) % 2 == 0,  # True for alternating blocks
                }
            )
            id_counter += 1

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326").set_index("block_id")

    return gdf
