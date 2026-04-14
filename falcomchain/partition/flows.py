import collections
import functools
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple


@dataclass
class Flow:
    """
    Groups the three incremental flow dictionaries computed between successive
    Partition steps. Avoids recomputing the full partition from scratch each step.

    :ivar node_flows: Maps each part to {"in": set of nodes that joined, "out": set that left}.
    :ivar part_flows: {"in": set of new district IDs, "out": set of removed district IDs}.
    :ivar candidate_flows: Maps each part to {"in": set of candidate nodes that joined, "out": set that left}.
    """

    node_flows: Optional[Dict]
    part_flows: Dict
    candidate_flows: Optional[Dict]

    @classmethod
    def initial(cls, flip) -> "Flow":
        """Flow for the very first partition (no parent)."""
        return cls(
            node_flows=None,
            part_flows={"in": set(flip.new_ids), "out": set()},
            candidate_flows=None,
        )

    @classmethod
    def from_parent(cls, parent, new_partition, superflip, flip) -> "Flow":
        """Compute all three flows from parent → new_partition in one call."""
        part_flows = compute_part_flows(superflip.merged_ids, flip.new_ids)
        node_flows = compute_node_flows(parent, new_partition)
        candidate_flows = _compute_candidate_flows(node_flows, new_partition.graph)
        return cls(node_flows=node_flows, part_flows=part_flows, candidate_flows=candidate_flows)


@functools.lru_cache(maxsize=2)
def neighbor_flips(partition) -> Set[Tuple]:
    """
    :param partition: A partition of a Graph
    :type partition: :class:`~falcomchain.partition.Partition`

    :returns: The set of edges that were flipped in the given partition.
    :rtype: Set[Tuple]
    """
    return {
        tuple(sorted((node, neighbor)))
        for node in partition.flip.flips
        for neighbor in partition.graph.neighbors(node)
        if neighbor not in partition.flow.part_flows["out"]
    }


def create_flow():
    return {"in": set(), "out": set()}


def compute_part_flows(merged_parts, new_ids):
    outgoing_ids = merged_parts.difference(
        new_ids
    )  # keys that will be removed from partition.parts.keys()
    incoming_ids = new_ids.difference(
        merged_parts
    )  # will be added to partition.parts.keys()
    part_flows = {"in": set(incoming_ids), "out": set(outgoing_ids)}
    return part_flows


def on_flow(initializer: Callable, alias: str) -> Callable:
    """
    Use this decorator to create an updater that responds to flows of nodes
    between parts of the partition.

    Decorate a function that takes:
    - The partition
    - The previous value of the updater on a fixed part P_i
    - The new nodes that are just joining P_i at this step
    - The old nodes that are just leaving P_i at this step
    and returns:
    - The new value of the updater for the fixed part P_i.

    This will create an updater whose values are dictionaries of the
    form `{part: <value of the given function on the part>}`.

    The initializer, by contrast, should take the entire partition and
    return the entire `{part: <value>}` dictionary.

    Example:

    .. code-block:: python

        @on_flow(initializer, alias='my_updater')
        def my_updater(partition, previous, new_nodes, old_nodes):
            # return new value for the part

    :param initializer: A function that takes the partition and returns a
        dictionary of the form `{part: <value>}`.
    :type initializer: Callable
    :param alias: The name of the updater to be created.
    :type alias: str

    :returns: A decorator that takes a function as input and returns a
        wrapped function.
    :rtype: Callable
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapped(partition, previous=None):
            if partition.parent is None:
                return initializer(partition)

            if previous is None:
                previous = partition.parent[alias]

            new_values = previous.copy()

            for part in partition.flow.part_flows["in"]:
                new_values[part] = set()

            for part, node_flow in partition.flow.node_flows.items():
                new_values[part] = function(
                    partition, previous[part], node_flow["in"], node_flow["out"]
                )

            for part in partition.flow.part_flows["out"]:
                new_values.pop(part, None)

            return new_values

        return wrapped

    return decorator


@functools.lru_cache(maxsize=2)
def compute_node_flows(old_partition, new_partition) -> Dict:
    """
    :param old_partition: A partition of a Graph representing dz    the previous step.
    :type old_partition: :class:`~falcomchain.partition.Partition`
    :param new_partition: A partition of a Graph representing the current step.
    :type new_partition: :class:`~falcomchain.partition.Partition`

    :returns: A dictionary mapping each node that changed assignment between
        the previous and current partitions to a dictionary of the form
        `{'in': <set of nodes that flowed in>, 'out': <set of nodes that flowed out>}`.
    :rtype: Dict
    """
    node_flows = collections.defaultdict(create_flow)

    for node, target in new_partition.flip.flips.items():
        source = old_partition.assignment.mapping[node]
        if source != target:
            node_flows[target]["in"].add(node)
            node_flows[source]["out"].add(node)
    return node_flows


def _compute_candidate_flows(node_flows: Dict, graph) -> Dict:
    candidate_flows = collections.defaultdict(create_flow)
    for part, flow in node_flows.items():
        candidate_flows[part] = {
            "in": {node for node in flow["in"] if graph.nodes[node]["candidate"] == 1},
            "out": {node for node in flow["out"] if graph.nodes[node]["candidate"] == 1},
        }
    return candidate_flows


class WrongFunction(Exception):
    """Raised when an unused function is called."""
