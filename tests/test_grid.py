import functools
import random
from typing import Optional
from unittest.mock import patch

import networkx
import pytest

from falcomchain.graph import Graph
from falcomchain.graph import Grid
from falcomchain.tree.tree import _part_nodes


def test_grid():

    grid = Grid(
        dimensions=(6, 6), num_candidates=32, density="corners", threshold=(2, 2)
    )
    graph = grid.graph

    nodes = {(i, j) for i in range(6) for j in range(6)}
    corners = {(0, 0), (0, 5), (5, 5), (5, 0)}
    sides = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
    }
    middle_nodes = nodes - (corners.union(sides))

    nodes_with_pop_70 = {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (4, 4),
        (4, 5),
        (5, 4),
        (5, 5),
        (0, 4),
        (0, 5),
        (1, 4),
        (1, 5),
        (4, 0),
        (5, 0),
        (4, 1),
        (5, 1),
    }
    nodes_with_pop_30 = set(graph.nodes) - nodes_with_pop_70

    assert grid.density == "corners"
    assert grid.num_candidates == 32
    assert type(graph) == Graph
    assert set(graph.nodes) == nodes
    assert all(graph.nodes[node]["area"] == 1 for node in graph.nodes)

    assert {
        node for node in graph.nodes if graph.nodes[node]["demand"] == 70
    } == nodes_with_pop_70
    assert {
        node for node in graph.nodes if graph.nodes[node]["demand"] == 30
    } == nodes_with_pop_30
    assert all(graph.nodes[node]["C_X"] == node[0] for node in graph.nodes)
    assert all(graph.nodes[node]["C_Y"] == node[1] for node in graph.nodes)
    assert all(graph.nodes[node]["boundary_node"] == True for node in corners and sides)
    assert all(graph.nodes[node]["boundary_node"] == False for node in middle_nodes)
    assert all(graph.nodes[node]["boundary_perim"] == 0 for node in middle_nodes)
    assert all(graph.nodes[node]["boundary_perim"] == 1 for node in sides)
    assert all(graph.nodes[node]["boundary_perim"] == 2 for node in corners)
    assert (
        sum(1 for node in graph.nodes if graph.nodes[node]["candidate"] == True)
        == grid.num_candidates
    )
