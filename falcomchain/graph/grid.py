"""
This module provides a Grid class used for creating and manipulating grid partitions.
It's part of the GerryChain suite, designed to facilitate experiments with redistricting
plans without the need for extensive data processing. This module relies on NetworkX for
graph operations and integrates with GerryChain's Partition class.

Dependencies:

- math: For math.floor() function.
- networkx: For graph operations with using the graph structure in
    :class:`~falcomchain.graph.Graph`.
- typing: Used for type hints.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import networkx

from .graph import Graph
from falcomchain.markovchain import polsby_popper
from falcomchain.partition import Partition
from falcomchain.random import rng


class Grid:
    """
    Synthetic grid graph generator for testing and demonstrations.

    .. note::
       This is a **testing/demo utility**, not a primary entry point.
       For production use cases, build your graph with
       :meth:`Graph.from_geodataframe` (geographic data) or
       :meth:`Graph.from_data` (raw data).

    Creates an m x n grid graph with all required FalcomChain node attributes
    (demand, area, C_X, C_Y, candidate) plus boundary information.

    Example usage::

        grid = Grid(dimensions=(10, 10), num_candidates=20, density="uniform")
        graph = grid.graph  # the underlying Graph object

    :param dimensions: (rows, cols) grid size.
    :param num_candidates: Number of nodes randomly selected as facility candidates.
    :param density: Demand pattern: 'uniform', 'opposite', or 'corners'.
    :param threshold: For non-uniform density, threshold tuple.
    :param candidate_ignore: Optional region to exclude from candidate sampling.

    Node attributes set: demand, area, C_X, C_Y, candidate, boundary_node, boundary_perim.
    Edge attributes set: shared_perim.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int],
        num_candidates: int,
        density: str,
        threshold: Optional[tuple] = None,
        candidate_ignore: Optional[int] = None,
    ) -> None:
        """
        :param dimensions: The grid dimensions (rows, columns), defaults to None.
        :type dimensions: Tuple[int, int], optional
        :param num_candidates:
        :type num_candidates:
        :param density: receives one of 'uniform', 'opposite', 'corners'.
        :type density: string
        :param candidate_ignore: a value of (x_0,y_0). Any node (x, y) with x < x_0 or y < y_0 will not be a candidate.
        :type candidate_ignore: tuple

        :raises Exception: If neither dimensions nor parent is provided.
        """
        if len(dimensions) != 2:
            raise Exception("Dimension must be 2.")

        self.density = density
        self.graph = self.create_grid_graph(dimensions)
        self.num_candidates = num_candidates
        self.candidate_ignore = candidate_ignore

        self.assign_coordinates()
        self.assign_candidates()
        self.tag_boundary_nodes(dimensions)
        self.get_boundary_perim(dimensions)

        if self.density != "uniform":
            self.assign_population(dimensions, threshold)

        # final step
        self.graph = Graph.from_networkx(self.graph)  # convert graph into Graph object

    # Main function which creates a grid graph with required node and edge attributes
    def create_grid_graph(self, dimensions: tuple) -> Graph:
        """
        Creates a grid graph with the specified dimensions.
        Optionally includes diagonal connections between nodes.

        :param dimensions: The grid dimensions (rows, columns).
        :type dimensions: Tuple[int, int]
        :param with_diagonals: If True, includes diagonal connections.
        :type with_diagonals: bool

        :returns: A grid graph.
        :rtype: Graph

        :raises ValueError: If the dimensions are not a tuple of length 2.
        """
        m, n = dimensions
        graph = networkx.generators.lattice.grid_2d_graph(m, n)

        networkx.set_edge_attributes(graph, 1, "shared_perim")

        networkx.set_node_attributes(graph, 50, "demand")
        networkx.set_node_attributes(graph, 1, "C_X")
        networkx.set_node_attributes(graph, 1, "C_Y")
        networkx.set_node_attributes(graph, 1, "area")

        values = {node: False for node in graph.nodes}
        networkx.set_node_attributes(graph, values, name="candidate")

        return graph

    def assign_coordinates(self) -> None:
        """
        Sets the specified attribute to the specified value for all nodes in the graph.

        :param graph: The graph to modify.
        :type graph: Graph
        :param attribute: The attribute to set.
        :type attribute: Any
        :param value: The value to set the attribute to.
        :type value: Any

        :returns: None
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["C_X"] = node[0]
            self.graph.nodes[node]["C_Y"] = node[1]

    def assign_candidates(self) -> None:
        "Sets self.num_candidates many nodes as candidates uniformly random on permitted region"
        nodes = set(self.graph.nodes)

        if self.candidate_ignore != None:
            x_0, y_0 = self.candidate_ignore
            ignore = {node for node in nodes if node[0] < x_0 or node[1] < y_0}
            nodes = nodes - ignore

        candidates = rng.sample(list(nodes), k=self.num_candidates)

        for node in self.graph.nodes:
            if node in candidates:
                self.graph.nodes[node]["candidate"] = True
            else:
                self.graph.nodes[node]["candidate"] = False

    def tag_boundary_nodes(self, dimensions: tuple) -> None:
        """
        Adds the boolean attribute ``boundary_node`` to each node in the graph.
        If the node is on the boundary of the grid, that node also gets the attribute
        ``boundary_perim`` which is determined by the function :func:`get_boundary_perim`.

        :param graph: The graph to modify.
        :type graph: Graph
        :param dimensions: The dimensions of the grid.
        :type dimensions: Tuple[int, int]

        :returns: None
        """
        m, n = dimensions
        for node in self.graph.nodes:
            if node[0] in {0, m - 1} or node[1] in {0, n - 1}:
                self.graph.nodes[node]["boundary_node"] = True
            else:
                self.graph.nodes[node]["boundary_node"] = False

    def get_boundary_perim(self, dimensions: tuple) -> int:  # this is wrong and useless
        """
        Determines the boundary perimeter of a node on the grid.
        The boundary perimeter is the number of sides of the node that
        are on the boundary of the grid.

        :param node: The node to check.
        :type node: Tuple[int, int]
        :param dimensions: The dimensions of the grid.
        :type dimensions: Tuple[int, int]

        :returns: The boundary perimeter of the node.
        :rtype: int
        """
        m, n = dimensions
        corners = {(0, 0), (m - 1, 0), (0, n - 1), (m - 1, n - 1)}
        middle = {
            node
            for node in self.graph.nodes
            if 0 < node[0] < m - 1 and 0 < node[1] < n - 1
        }
        sides = set(self.graph.nodes) - (corners.union(middle))

        for node in corners:
            self.graph.nodes[node]["boundary_perim"] = 2

        for node in middle:
            self.graph.nodes[node]["boundary_perim"] = 0

        for node in sides:
            self.graph.nodes[node]["boundary_perim"] = 1

    def assign_population(self, dimensions: tuple, threshold: tuple) -> int:
        """
        Assigns a color (as an integer) to a node based on its x-coordinate.

        This function is used to partition the grid into two parts based on a given threshold.
        Nodes with an x-coordinate less than or equal to the threshold are assigned one color,
        and nodes with an x-coordinate greater than the threshold are assigned another.

        :param node: The node to color, represented as a tuple of coordinates (x, y).
        :type node: Tuple[int, int]
        :param threshold: The x-coordinate value that determines the color assignment.
        :type threshold: int

        :returns: An integer representing the color of the node. Returns 0 for nodes with
            x-coordinate less than or equal to the threshold, and 1 otherwise.
        :rtype: int
        """

        if self.density == "opposite":
            for node in self.graph.nodes:
                x, y = node
                if x >= threshold[0] and y >= threshold[1]:
                    self.graph.nodes[node]["demand"] = 70
                elif x < threshold[0] and y < threshold[1]:
                    self.graph.nodes[node]["demand"] = 70
                else:
                    self.graph.nodes[node]["demand"] = 30

        if self.density == "corners":
            k_1, k_2 = threshold
            m, n = dimensions
            for node in self.graph.nodes:
                x, y = node
                if k_1 <= x < m - k_1 or k_2 <= y < n - k_2:
                    self.graph.nodes[node]["demand"] = 30
                else:
                    self.graph.nodes[node]["demand"] = 70
