from collections import defaultdict
from collections.abc import Mapping
from typing import DefaultDict, Dict, Optional, Set, Tuple, Union

import pandas

from falcomchain.graph import Graph


class FacilityError(Exception):
    "raised in facility assignment of Assignment"


class Assignment(Mapping):
    """
    An assignment of nodes into parts.

    The goal of :class:`Assignment` is to provide an interface that mirrors a
    dictionary (what we have been using for assigning nodes to districts) while making it
    convenient/cheap to access the set of nodes in each part.

    An :class:`Assignment` has a ``parts`` property that is a dictionary of the form
    ``{part: <frozenset of nodes in part>}``.
    """

    __slots__ = ["parts", "mapping", "candidates", "teams"]

    travel_times = None

    def __init__(
        self,
        parts: Dict,
        candidates: Dict,
        teams: Dict,
        mapping: Optional[Dict] = None,
        validate: bool = True,
    ) -> None:
        """
        :param parts: Dictionary mapping partition assignments frozensets of nodes.
        :type parts: Dict
        :param centers:
        :type centers: Dict
        :param radius:
        :type radius: Dict
        :param candidates:
        :type candidates: Dict
        param teams:
        :type teams: Dict
        :param mapping: Dictionary mapping nodes to partition assignments. Default is None.
        :type mapping: Optional[Dict], optional
        :param validate: Whether to validate the assignment. Default is True.
        :type validate: bool, optional

        :returns: None

        :raises ValueError: if the keys of ``parts`` are not unique
        :raises TypeError: if the values of ``parts`` are not frozensets
        """
        if validate:
            number_of_keys = sum(len(keys) for keys in parts.values())
            number_of_unique_keys = len(set().union(*parts.values()))
            if number_of_keys != number_of_unique_keys:
                raise ValueError("Keys must have unique assignments.")
            if not all(isinstance(keys, frozenset) for keys in parts.values()):
                raise TypeError("Level sets must be frozensets")

        self.parts = parts
        self.candidates = candidates
        self.teams = teams

        if not mapping:
            self.mapping = {}
            for part, nodes in self.parts.items():
                for node in nodes:
                    self.mapping[node] = part
        else:
            self.mapping = mapping

    def __repr__(self):
        return "<Assignment [{} keys, {} parts]>".format(len(self), len(self.parts))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(len(keys) for keys in self.parts.values())

    def __getitem__(self, node):
        return self.mapping[node]

    def copy(self) -> "Assignment":
        """
        Returns a copy of the assignment.
        Does not duplicate the frozensets of nodes, just the parts dictionary.
        """
        return Assignment(
            self.parts.copy(),
            self.candidates.copy(),
            self.teams.copy(),
            self.mapping.copy(),
            validate=False,
        )

    def facility_assignment(self, part) -> tuple:
        """
        Find the best facility center for ``part`` by minimising the maximum
        travel time from the candidate to any node in the part (minimax radius).

        :returns: ``(best_candidate, radius)``
        """
        travel_times = self.travel_times
        best_candidate = None
        best_radius = float("inf")

        for candidate in self.candidates[part]:
            radius = max(travel_times[(candidate, node)] for node in self.parts[part])
            if radius < best_radius:
                best_radius = radius
                best_candidate = candidate

        return best_candidate, best_radius

    # instead of iterating over parts again, we can do it in update_flows function
    def update_part_flows(self, part_flows):

        for part in part_flows["in"]:
            self.parts[part] = set()
            self.candidates[part] = set()
            self.teams[part] = None

        for part in part_flows["out"]:
            self.parts.pop(part, None)
            self.candidates.pop(part, None)
            self.teams.pop(part, None)

    def update_flows(self, flow, team_flips):
        """
        Update the assignment using the given Flow object.

        :param flow: The Flow computed from the parent partition.
        :type flow: :class:`~falcomchain.partition.flows.Flow`
        :param team_flips: Maps district IDs to updated team counts.
        :type team_flips: Dict
        """
        self.update_part_flows(flow.part_flows)

        for part, node_flow in flow.node_flows.items():
            if part not in flow.part_flows["out"]:
                self.parts[part] = frozenset(
                    (self.parts[part] - node_flow["out"]) | node_flow["in"]
                )

                for node in node_flow["in"]:
                    self.mapping[node] = part

                self.teams[part] = team_flips[part]

                cand_flow = flow.candidate_flows[part]
                self.candidates[part] = frozenset(
                    (self.candidates[part] - cand_flow["out"]) | cand_flow["in"]
                )

        if len(self.mapping) != sum(len(self.parts[part]) for part in self.parts):
            parts_from_mapping = {part: set() for part in self.mapping.values()}
            for key, part in self.mapping.items():
                parts_from_mapping[part].add(key)

            raise Exception(
                "mapping does not match parts.\n"
                f"part flows {flow.part_flows} \n"
                f"node flows {flow.node_flows} \n"
                f"mapping {self.mapping} \n"
                f"parts {self.parts}"
            )

    def items(self):
        """
        Iterate over ``(node, part)`` tuples, where ``node`` is assigned to ``part``.
        """
        yield from self.mapping.items()

    def keys(self):
        yield from self.mapping.keys()

    def values(self):
        yield from self.mapping.values()

    def to_series(self) -> pandas.Series:
        """
        :returns: The assignment as a :class:`pandas.Series`.
        :rtype: pandas.Series
        """
        groups = [
            pandas.Series(data=part, index=nodes) for part, nodes in self.parts.items()
        ]
        return pandas.concat(groups)

    def to_dict(self) -> Dict:
        """
        :returns: The assignment as a ``{node: part}`` dictionary.
        :rtype: Dict
        """
        return self.mapping

    @classmethod
    def from_dict(cls, assignment: Dict, graph: Graph, teams: Dict) -> "Assignment":
        """
        Create an :class:`Assignment` from a dictionary. This is probably the method you want
        to use to create a new assignment.

        This also works for :class:`pandas.Series`.

        :param assignment: dictionary mapping nodes to partition assignments
        :type assignment: Dict

        :returns: A new instance of :class:`Assignment` with the same assignments as the
            passed-in dictionary.
        :rtype: Assignment
        """
        sets, facilities = level_sets(assignment, graph)
        parts = {part: frozenset(keys) for part, keys in sets.items()}
        candidates = {part: frozenset(keys) for part, keys in facilities.items()}
        return cls(parts, candidates, teams)


def get_assignment(
    part_assignment: Dict,
    graph: Graph,
    teams: Dict,
) -> Assignment:
    """
    Either extracts an :class:`Assignment` object from the input graph
    using the provided key or attempts to convert part_assignment into
    an :class:`Assignment` object.

    :param part_assignment: A node attribute key, dictionary, or
        :class:`Assignment` object corresponding to the desired assignment.
    :type part_assignment: str
    :param graph: The graph from which to extract the assignment.
        Default is None.
    :type graph: Optional[Graph], optional

    :returns: An :class:`Assignment` object containing the assignment
        corresponding to the part_assignment input
    :rtype: Assignment

    :raises TypeError: If the part_assignment is a string and the graph
        is not provided.
    :raises TypeError: If the part_assignment is not a string or dictionary.
    """
    # if isinstance(part_assignment, str):
    #    if graph is None:
    #        raise TypeError(
    #            "You must provide a graph when using a node attribute for the part_assignment"
    #        )
    #    return Assignment.from_dict(
    #        {node: graph.nodes[node][part_assignment] for node in graph}
    #    )
    # Check if assignment is a dict or a mapping type
    # elif callable(getattr(part_assignment, "items", None)):
    return Assignment.from_dict(part_assignment, graph, teams)
    # elif isinstance(part_assignment, Assignment):
    #    return part_assignment
    # else:
    #    raise TypeError("Assignment must be a dict or a node attribute key")


def level_sets(
    assignment: dict, graph: Graph, container: type[Set] = set
) -> Tuple[Dict, Dict]:
    """
    Inverts a dictionary. ``{key: value}`` becomes
    ``{value: <container of keys that map to value>}``.

    :param mapping: A dictionary to invert. Keys and values can be of any type.
    :type mapping: Dict
    :param container: A container type used to collect keys that map to the same value.
        By default, the container type is ``set``.
    :type container: Type[Set], optional

    :return: A dictionary where each key is a value from the original dictionary,
        and the corresponding value is a container (by default, a set) of keys from
        the original dictionary that mapped to this value.
    :rtype: DefaultDict

    Example usage::

    .. code_block:: python

        >>> level_sets({'a': 1, 'b': 1, 'c': 2})
        defaultdict(<class 'set'>, {1: {'a', 'b'}, 2: {'c'}})
    """
    sets: Dict = defaultdict(container)
    candidates: Dict = defaultdict(container)

    for node, part in assignment.items():
        sets[part].add(node)
        if graph.nodes[node]["candidate"] == 1:
            candidates[part].add(node)

    for part in sets.keys():
        if not any(graph.nodes[node]["candidate"] == 1 for node in sets[part]):
            print(f"Part {part} does not have a candidate.")
            print(sets[part])

    return sets, candidates
