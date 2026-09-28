from collections.abc import Mapping
from types import SimpleNamespace

import networkx as nx
import pandas
import pytest

from falcomchain.partition.assignment import Assignment, get_assignment


@pytest.fixture
def graph():
    """Path 1-2-3-4 with candidates at nodes 1 and 3."""
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4)])
    for n in g.nodes:
        g.nodes[n]["candidate"] = 1 if n in (1, 3) else 0
        g.nodes[n]["demand"] = 10
    return g


@pytest.fixture
def assignment(graph):
    return Assignment.from_dict({1: 1, 2: 2, 3: 2, 4: 2}, graph, teams={1: 1, 2: 1})


class TestAssignment:
    def test_parts_candidates_and_teams(self, assignment):
        assert assignment.parts == {1: frozenset({1}), 2: frozenset({2, 3, 4})}
        assert assignment.candidates == {1: frozenset({1}), 2: frozenset({3})}
        assert assignment.teams == {1: 1, 2: 1}

    def test_implements_Mapping_abc(self, assignment):
        assert isinstance(assignment, Mapping)
        assert len(assignment) == 4
        assert set(assignment) == {1, 2, 3, 4}
        assert assignment[1] == 1 and assignment[3] == 2
        assert set(assignment.keys()) == {1, 2, 3, 4}
        assert set(assignment.values()) == {1, 2}
        assert set(assignment.items()) == {(1, 1), (2, 2), (3, 2), (4, 2)}
        assert assignment == {1: 1, 2: 2, 3: 2, 4: 2}

    def test_has_get_method_like_a_dict(self, assignment):
        assert assignment.get(1) == 1
        assert assignment.get("not a node", 5) == 5

    def test_raises_keyerror_for_missing_nodes(self, assignment):
        with pytest.raises(KeyError):
            assignment["not a node"]

    def test_to_series_and_to_dict(self, assignment):
        series = assignment.to_series()
        assert isinstance(series, pandas.Series)
        assert dict(series.items()) == {1: 1, 2: 2, 3: 2, 4: 2}
        assert assignment.to_dict() == {1: 1, 2: 2, 3: 2, 4: 2}

    def test_copy_shares_the_node_sets(self, assignment):
        copy = assignment.copy()
        assert copy == assignment
        for part in assignment.parts:
            assert copy.parts[part] is assignment.parts[part]
        assert copy.parts is not assignment.parts

    def test_from_series(self, graph):
        series = pandas.Series([1, 2, 2, 2], index=[1, 2, 3, 4])
        assignment = Assignment.from_dict(series, graph, teams={1: 1, 2: 1})
        assert assignment == {1: 1, 2: 2, 3: 2, 4: 2}

    def test_raises_if_a_node_has_two_assignments(self):
        with pytest.raises(ValueError):
            Assignment({"one": frozenset({1, 2, 3}), "two": frozenset({1, 4, 5})},
                       candidates={}, teams={})

    def test_raises_if_parts_are_not_frozensets(self):
        with pytest.raises(TypeError):
            Assignment({"one": {1, 2}}, candidates={}, teams={})

    def test_update_flows_moves_nodes_and_teams(self, assignment):
        # Node 2 moves from part 2 to part 1; part 1 keeps its candidate.
        flow = SimpleNamespace(
            part_flows={"in": set(), "out": set()},
            node_flows={1: {"in": {2}, "out": set()}, 2: {"in": set(), "out": {2}}},
            candidate_flows={1: {"in": set(), "out": set()},
                             2: {"in": set(), "out": set()}},
        )
        assignment.update_flows(flow, team_flips={1: 2, 2: 1})
        assert assignment[2] == 1
        assert assignment.parts[1] == frozenset({1, 2})
        assert assignment.parts[2] == frozenset({3, 4})
        assert assignment.teams == {1: 2, 2: 1}

    def test_update_flows_adds_and_removes_parts(self, assignment):
        # Part 2 is dissolved into a brand-new part 3 (its nodes and candidate move).
        flow = SimpleNamespace(
            part_flows={"in": {3}, "out": {2}},
            node_flows={3: {"in": {2, 3, 4}, "out": set()}},
            candidate_flows={3: {"in": {3}, "out": set()}},
        )
        assignment.update_flows(flow, team_flips={1: 1, 3: 1})
        assert set(assignment.parts) == {1, 3}
        assert assignment.parts[3] == frozenset({2, 3, 4})
        assert assignment.candidates[3] == frozenset({3})
        assert assignment == {1: 1, 2: 3, 3: 3, 4: 3}


def test_get_assignment_builds_an_assignment(graph):
    assignment = get_assignment({1: 1, 2: 2, 3: 2, 4: 2}, graph, teams={1: 1, 2: 1})
    assert isinstance(assignment, Assignment)
    assert assignment == {1: 1, 2: 2, 3: 2, 4: 2}


def test_repr(assignment):
    assert repr(assignment) == "<Assignment [4 keys, 2 parts]>"
