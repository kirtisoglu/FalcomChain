from collections import namedtuple
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import networkx as nx

from falcomchain.graph import FrozenGraph, Graph
from falcomchain.helper import load_pickle, save_pickle
from falcomchain.tree.tree import Flip, capacitated_recursive_tree

from .assignment import Assignment, get_assignment
from .flows import Flow, neighbor_flips
from .subgraphs import SubgraphView


class Partition:
    """
    Partition represents a partition of the nodes of the graph. It will perform
    the first layer of computations at each step in the Markov chain - basic
    aggregations and calculations that we want to optimize.

    :ivar graph: The underlying graph.
    :type graph: :class:`~falcomchain.graph.Graph`
    :ivar assignment: Maps node IDs to district IDs.
    :type assignment: :class:`~falcomchain.partition.assignment.Assignment`
    :ivar parts: Maps district IDs to the set of nodes in that district.
    :type parts: Dict
    :ivar subgraphs: Maps district IDs to the induced subgraph of that district.
    :type subgraphs: Dict
    """

    __slots__ = (
        "graph",
        "capacity_level",
        "subgraphs",
        "supergraph",
        "assignment",
        "parent",
        "superflip",
        "flip",
        "flow",
        "step",
    )


    def __init__(
        self,
        capacity_level: Optional[int] = None, 
        graph=None, 
        flip=None,
        superflip=None,
        parent=None,
        assignment=None, # ?
    ):
        """
        :param graph: Underlying graph.
        :param assignment: Dictionary assigning nodes to districts.
        :param updaters: Dictionary of functions to track data about the partition.
            The keys are stored as attributes on the partition class,
            which the functions compute.
        :param use_default_updaters: If `False`, do not include default updaters.
        """
    
        
        if parent is None:
            self._first_time(
                graph,
                assignment,  
                #updaters,
                #use_default_updaters,
                capacity_level,
                flip
            )
        else:
            self._from_parent(parent, flip, superflip)

        # define here if it is not needed to be defined before _from_parent()
        #self._cache = dict()
        self.subgraphs = SubgraphView(self.graph, self.parts)

    @classmethod
    def from_random_assignment(
        cls,
        graph: Graph,
        epsilon: float,
        demand_target: int,
        assignment_class: Assignment,
        #updaters: Optional[Dict[str, Callable]] = None,
        #use_default_updaters: bool = True,
        capacity_level=1,
        density: Optional[float] = None,
    ) -> "Partition":
        """
        Create a Partition with a random assignment of nodes to districts.

        :param graph: The graph to create the Partition from.
        :type graph: :class:`~falcomchain.graph.Graph`
        :param teams:The total of number of doctor-nurse teams to hire at centers
        :type teams: int
        :param capacity_level: The maximum number of doctor nurse teams at a facility
        :type capacity_level: int
        :param epsilon: The maximum relative population deviation from the ideal
        :type epsilon: float
            population. Should be in [0,1].
        :param demand_col: The column of the graph's node data that holds the demand data.
        :type demand_col: str
        :param updaters: Dictionary of updaters
        :type updaters: Optional[Dict[str, Callable]], optional
        :param use_default_updaters: If `False`, do not include default updaters.
        :type use_default_updaters: bool, optional
        :param method: The function to use to partition the graph into ``n_parts``. Defaults to
            :func:`~falcomchain.tree.capacitated_recursive_tree`.
        :type method: Callable, optional

        :returns: The partition created with a random assignment
        :rtype: Partition
        """
        total_pop = sum(graph.nodes[n]["demand"] for n in graph)
        n_teams = int(total_pop // demand_target)
        # if capacity_level is 1, n_teams becomes number of districts.

        flip = capacitated_recursive_tree(
            graph=graph,
            n_teams=n_teams,
            demand_target=demand_target,
            epsilon=epsilon,
            capacity_level=capacity_level,
            density=density,
        )

        return cls(
            capacity_level=capacity_level,
            assignment=flip.flips,
            #updaters=updaters,
            #use_default_updaters=use_default_updaters,
            graph=graph,
            flip= flip
        )

    def _first_time(
        self,
        graph,
        assignment,  
        #updaters,
        #use_default_updaters,
        capacity_level,
        flip,
    ):
        if isinstance(graph, Graph):
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, networkx.Graph):
            graph = Graph.from_networkx(graph)
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, FrozenGraph):
            self.graph = graph
        else:
            raise TypeError(f"Unsupported Graph object with type {type(graph)}")


        self.step = 1
        self.parent = None
        self.capacity_level = capacity_level
        
        self.flip = flip
        self.superflip = None
        self.flow = Flow.initial(flip)
        self.assignment = get_assignment(assignment, graph, flip.team_flips)

        #if updaters is None:
        #    updaters = {}

        #if use_default_updaters:
        #    self.updaters = self.default_updaters.copy()  # copy
        #else:
        #    self.updaters = {}
        #self.updaters.update(updaters)
        
        #self.cut_edges = cut_edges(self)
        self.supergraph = supergraph(self)


    def _from_parent(
        self,
        parent: "Partition",
        flip: Flip,
        superflip: Flip,
    ) -> None:

        self.step = parent.step + 1
        self.parent = parent
        self.graph = parent.graph
        self.capacity_level = parent.capacity_level
        #self.updaters = parent.updaters.copy()
        self.flip = flip
        self.superflip = superflip
        self.flow = Flow.from_parent(parent, self, superflip, flip)
        self.assignment = parent.assignment.copy()
        self.assignment.update_flows(self.flow, self.flip.team_flips)
        
        #self.cut_edges = cut_edges(self) # done
        self.supergraph = supergraph(self) # done
        

    def __repr__(self):
        number_of_parts = len(self)
        s = "s" if number_of_parts > 1 else ""
        return "<{} [{} part{}]>".format(self.__class__.__name__, number_of_parts, s)

    def __len__(self):
        return len(self.parts)


    def perform_flip(self, flipp: Flip, superflipp: Flip) -> "Partition":
        """
        Returns the new partition obtained by performing the given `flips` and new_teams.
        on this partition.
        :param flip: 
        :param superflip: 
        :returns: the new :class:`Partition`
        """
        return self.__class__(
            parent=self,
            flip = flipp,
            superflip=superflipp,
        )


    def crosses_parts(self, edge: Tuple) -> bool:
        """
        :param edge: tuple of node IDs
        :type edge: Tuple

        :returns: True if the edge crosses from one part of the partition to another
        :rtype: bool
        """
        return self.assignment.mapping[edge[0]] != self.assignment.mapping[edge[1]]

    def part_demand(self, part):
        return sum(self.graph.nodes[node]["demand"] for node in self.parts[part])
    
    
    def part_area(self, part):
        return sum(self.graph.nodes[node]["area"] for node in self.parts[part])


    @property
    def parts(self):
        return self.assignment.parts

    @property
    def teams(self):
        return self.assignment.teams

    @property
    def candidates(self):
        return self.assignment.candidates


    def save(self, path: str):
        """
        Serializes the partition's assignment, team allocations, and metadata to a pickle file.

        :param path: File path to write to.
        :type path: str
        """
        flips = self.assignment.mapping
        teams = self.teams
        metadata = {"capacity_level": self.capacity_level}
        data = {"flips": flips, "team_flips": teams, "metadata": metadata}
        save_pickle(data, path)

    def save_json(self, path: str):
        """
        Save the partition as a human-readable JSON file.

        The file contains the full assignment (node -> district), team
        allocations (district -> teams), per-district demand, and metadata.
        Node and district IDs are converted to strings for JSON compatibility.

        :param path: File path to write to.
        :type path: str
        """
        import json

        districts = {}
        for part_id, nodes in self.parts.items():
            demand = sum(self.graph.nodes[n].get("demand", 0) for n in nodes)
            candidates = [str(n) for n in nodes
                          if self.graph.nodes[n].get("candidate", 0)]
            districts[str(part_id)] = {
                "nodes": sorted(str(n) for n in nodes),
                "teams": self.teams[part_id],
                "demand": demand,
                "candidates": candidates,
            }

        data = {
            "num_districts": len(self.parts),
            "total_teams": sum(self.teams.values()),
            "total_nodes": sum(len(nodes) for nodes in self.parts.values()),
            "capacity_level": self.capacity_level,
            "assignment": {str(k): str(v) for k, v in self.assignment.mapping.items()},
            "districts": districts,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def write_to_graph(self, graph=None):
        """
        Write this partition's state into the graph as node and graph attributes.
        After this, the graph alone fully describes the partition.

        :param graph: Optional Graph (or networkx.Graph) to write to. Defaults
            to ``self.graph.graph`` (the underlying mutable graph).
        """
        if graph is None:
            graph = self.graph.graph if hasattr(self.graph, "graph") else self.graph

        for node, dist_id in self.assignment.mapping.items():
            if node in graph:
                graph.nodes[node]["district"] = dist_id

        graph.graph["teams_per_district"] = dict(self.teams)
        graph.graph["capacity_level"] = self.capacity_level

    @classmethod
    def from_graph(cls, graph) -> "Partition":
        """
        Construct a Partition from a graph that has ``district`` node attributes
        and ``teams_per_district`` graph attribute. This is the inverse of
        ``write_to_graph()``.

        :param graph: A Graph (or networkx.Graph) with partition state stored
            as attributes.
        :returns: A Partition instance.
        :rtype: Partition

        :raises ValueError: If the graph lacks the required ``district`` attribute.
        """
        g = graph.graph if hasattr(graph, "graph") and hasattr(graph.graph, "nodes") else graph

        sample_node = next(iter(g.nodes), None)
        if sample_node is None or "district" not in g.nodes[sample_node]:
            raise ValueError(
                "Graph has no 'district' node attribute. "
                "Use Partition.from_random_assignment() to create a fresh partition, "
                "or call partition.write_to_graph() before serializing."
            )

        assignment = {n: g.nodes[n]["district"] for n in g.nodes}
        teams = g.graph.get("teams_per_district", {})
        capacity_level = g.graph.get("capacity_level", 1)

        return cls(
            capacity_level=capacity_level,
            assignment=assignment,
            flip=Flip(
                flips=assignment,
                team_flips=teams,
                new_ids=frozenset(teams.keys()),
            ),
            graph=graph,
        )

    @classmethod
    def load_json(cls, json_path: str, graph):
        """
        Load a partition from a JSON file and a graph.

        :param json_path: Path to the JSON partition file.
        :type json_path: str
        :param graph: The graph this partition belongs to.
        :returns: A restored Partition instance.
        :rtype: Partition
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        # Reconstruct the assignment dict with original node types
        # by matching string keys back to graph nodes
        node_str_to_id = {str(n): n for n in graph.nodes}
        assignment = {}
        for node_str, part_str in data["assignment"].items():
            node_id = node_str_to_id.get(node_str, node_str)
            assignment[node_id] = int(part_str) if part_str.isdigit() else part_str

        team_flips = {
            int(k) if k.isdigit() else k: v["teams"]
            for k, v in data["districts"].items()
        }

        return cls(
            capacity_level=data["capacity_level"],
            assignment=assignment,
            flip=Flip(
                flips=assignment,
                team_flips=team_flips,
                new_ids=frozenset(team_flips.keys()),
            ),
            graph=graph,
        )

    @classmethod
    def load_partition(cls, graph_path: str, partition_path: str):
        """
        Loads a partition from saved graph and partition pickle files.

        :param graph_path: Path to the saved graph pickle file.
        :type graph_path: str
        :param partition_path: Path to the saved partition pickle file.
        :type partition_path: str
        :returns: A restored Partition instance.
        :rtype: Partition
        """
        my_graph = load_pickle(graph_path)
        partition = load_pickle(partition_path)

        return cls(
            capacity_level=partition["metadata"]["capacity_level"],
            assignment=partition["flips"],
            flip=Flip(
                flips=partition["flips"],
                team_flips=partition["team_flips"],
                new_ids=set(partition["team_flips"].keys()),
            ),
            graph=my_graph,
        )
    
    
    
class SupergraphError(Exception):
    """Raised when supergraph constructed wrong."""

    
def supergraph(partition:Partition):
    # Later, you can define this over superflips. 
    
    new_ones = set(partition.flip.new_ids.copy())
    
    if new_ones != set(partition.flip.flips.values()):
        raise SupergraphError(f"new ids do not match with flip values.\n"
                              f"new ids: {new_ones}\n"
                              f"flips values: {set(partition.flip.flips.values())}")
        
    
    # if flips are correct, then new ones are correct.
    
    # starts here
    if partition.parent==None:  # initial partition
        graph = nx.Graph()
        merged = set()
        
    else:
        merged =set(partition.superflip.merged_ids.copy())
        graph = partition.parent.supergraph.copy()
        graph.remove_nodes_from(list(merged))  # Edges has gone too.

    leaving = merged - new_ones
    if leaving != partition.flow.part_flows["out"]:
        raise SupergraphError(f"leaving parts {leaving} is not same as part out flow {partition.flow.part_flows['out']}")
    
    for node in leaving:
        if node in graph.nodes:
            raise SupergraphError(f"leaving node {node} is still here.")
            
            
    for new in new_ones:
        if new not in partition.parts.keys():
            raise SupergraphError(f"new_ones has an id that is not in partition.parts {new}")
        

    for part in leaving:
        if part in partition.parts:
            raise SupergraphError(f"part {part} is not in the partition parts")
        
        
    # ---- add nodes
    nodes = [(node, {"demand":partition.part_demand(node),
                     "area": partition.part_area(node), 
                     "n_teams":partition.flip.team_flips[node],
                     "n_candidates":len(partition.candidates[node])
                    }
              ) for node in new_ones]
    
    try:
        graph.add_nodes_from(nodes)
        
    except Exception:
        raise SupergraphError("couldn't add nodes to supergraph")


    # add edges
    add = {(node, neighbor) for node in partition.flip.flips 
            for neighbor in partition.graph.neighbors(node)
            #if partition.flip.flips[neighbor] not in leaving
            }

    for edge in add:
        u,v = edge
        uu, vv = partition.assignment.mapping[u], partition.assignment.mapping[v]
        
        if uu not in graph.nodes or vv not in graph.nodes:
            raise SupergraphError(f"one of endpoints {u,v} not in supergraph.\n"
                                  f"endpoints are {uu, vv}\n"
                                  f"leaving is {leaving}")

    try:
        add_edges = {
            (node, neighbor)
            for (node, neighbor) in add
            if partition.crosses_parts((node, neighbor))}
    
    except Exception:
        raise print(f"add edges {add_edges}")
    
    for edge in add_edges:
        uu = partition.assignment.mapping[edge[0]]
        vv = partition.assignment.mapping[edge[1]]
        graph.add_edge(uu,vv)

    return graph


