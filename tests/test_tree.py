import functools
import random
from typing import Optional
from unittest.mock import patch

import networkx
import pytest

from falcomchain.tree.tree import _part_nodes


def test_spanningtree(spanningtree_with_forced_root):

    #      1 - 2 - 3 - 4
    #          |       |
    #      5 - 6 - 7   8 - 9 - 10
    #      |       |       |
    #      11-12   13      14
    #      |
    #      15-16-17
    #      |
    #      18-19
    #      |
    #      20

    # candidates = {3, 7, 8, 10, 12, 17, 18}

    tree = spanningtree_with_forced_root

    assert tree.total_demand == 200
    assert tree.ideal_demand == 40
    assert tree.root == 2
    assert tree.graph.degree(tree.root) > 1
    assert tree.n_teams == 5

    successors = {
        2: [1, 3, 6],
        3: [4],
        6: [5, 7],
        4: [8],
        5: [11],
        7: [13],
        8: [9],
        11: [12, 15],
        9: [10, 14],
        15: [16, 18],
        16: [17],
        18: [19, 20],
    }
    assert tree.successors == successors

    # test the accumulation
    accumulated_candidates = {
        node for node in tree.graph.nodes if tree.graph.nodes[node]["candidate"] > 0
    }
    assert accumulated_candidates == {
        2,
        3,
        4,
        8,
        9,
        10,
        7,
        6,
        5,
        11,
        12,
        15,
        16,
        17,
        18,
    }

    nodes_11 = _part_nodes(successors=tree.successors, start=11)
    nodes_7 = _part_nodes(successors=tree.successors, start=7)
    assert nodes_11 == {11, 12, 15, 16, 17, 18, 19, 20}
    assert nodes_7 == {7, 13}

    """
    part_nodes for all nodes 
    
    {1: {1},
    2: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    3: {3, 4, 8, 9, 10, 14},
    4: {4, 8, 9, 10, 14},
    6: {5, 6, 7, 11, 12, 13, 15, 16, 17, 18, 19, 20},
    8: {8, 9, 10, 14},
    5: {5, 11, 12, 15, 16, 17, 18, 19, 20},
    7: {7, 13},
    9: {9, 10, 14},
    10: {10},
    11: {11, 12, 15, 16, 17, 18, 19, 20},
    13: {13},
    14: {14},
    12: {12},
    15: {15, 16, 17, 18, 19, 20},
    16: {16, 17},
    17: {17},
    18: {18, 19, 20},
    19: {19},
    20: {20}}
    
    """
