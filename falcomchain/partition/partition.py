import math
from collections import namedtuple
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import networkx as nx

from falcomchain.graph import FrozenGraph, Graph
from falcomchain.helper import load_pickle, save_pickle
from falcomchain.tree.tree import Flip, capacitated_recursive_tree

from .assignment import Assignment, get_assignment
from .flows import Flow, neighbor_flips
from .subgraphs import SubgraphView


def _init_unit_phi_psi(phi, gamma, r):
    """Cut-selection score used during initial-state generation only.

    Biases the recursive bipartition toward subtrees whose single
    candidate is **geographically central** to its subtree, i.e.:

      * ``phi ≈ 1`` (one candidate per subtree, the natural district
        shape — vs the original ``psi = phi`` which prefers subtrees
        that contain MANY candidates, draining the residual of
        candidate-anchors and causing the recursion to stall late on
        sparse-candidate graphs like LAS).
      * Low ``r`` (the candidate has small eccentricity within the
        subtree — its travel time to the furthest node in the subtree
        is short, so it can serve the whole district well).

    Combined formula::

        psi = (1 / phi) * exp(-gamma * r)

    At ``gamma == 0`` or when ``travel_times`` is unset, the centrality
    factor collapses to 1 and only the unit-phi bias remains. With
    ``travel_times`` and ``gamma > 0`` set during initial-state
    generation, cuts that produce well-shaped single-candidate
    districts are preferred.

    Used **only by** :meth:`Partition._from_random_global` and
    :meth:`Partition._from_random_within_zones`. Chain proposals
    (``hierarchical_recom``) keep the unbiased default so detailed
    balance arguments stand.

    :param phi: Number of candidates in the subtree.
    :param gamma: Candidate-awareness inverse temperature.
    :param r: Demand radius of the subtree (min eccentricity over
        candidates ∩ subtree).
    """
    if phi < 1:
        return 0.0
    score = 1.0 / phi
    if gamma > 0.0 and r is not None and r != float("inf"):
        import math
        score *= math.exp(-gamma * r)
    return score


# Default candidate-awareness for initial-state generation. A mild
# positive value (gamma_init = 1 / typical_travel_time) brings the
# centrality bias in without dominating the unit-phi bias.
_INIT_GAMMA = 0.1


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
        "super_assignment",
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
        super_assignment: Optional[Dict] = None,
        init_super_partition: bool = False,
        epsilon_super: Optional[float] = None,
        c_min: int = 1,
        c_min_super: int = 1,
        c_max_super: Optional[int] = None,
        min_districts_super: int = 1,
        rule: str = "per_team",
        enforce_global_balance: bool = False,
        psi_fn: Optional[Callable] = None,
        super_psi_fn: Optional[Callable] = None,
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
        :param super_assignment: Optional dict mapping each base node to a
            superdistrict ID (e.g. health-zone). When provided, level-1
            districts are formed by partitioning each superdistrict
            independently — every level-1 district lives entirely inside
            one superdistrict, and the resulting :attr:`super_assignment`
            on the Partition reflects this fixed grouping. When ``None``
            (default), the graph is partitioned globally and each level-1
            district becomes its own superdistrict (identity level-2).
        :param init_super_partition: When ``True`` (and ``super_assignment``
            is not provided), additionally call recursive partitioning on
            the supergraph to produce a non-trivial initial level-2
            partition (paper Section 5.1, "Initialization"). Default
            ``False`` keeps the cheap identity grouping. Ignored when
            ``super_assignment`` is provided (the user-supplied grouping
            is already a real level-2 partition).
        :param epsilon_super: Level-2 demand-balance tolerance ε² used by
            ``init_super_partition``. Defaults to ``epsilon`` when not set.
        :param c_min: Minimum capacity per district (paper's ``c^ℓ_min``).
            Default 1. With ``c_min > 1`` no district has fewer than
            ``c_min`` teams; the chain explores a smaller state space and
            Assumption 6.1 relaxes by a factor of ``c_min``.

        :returns: The partition created with a random assignment
        :rtype: Partition
        """
        if super_assignment is not None:
            return cls._from_random_within_zones(
                graph, epsilon, demand_target, capacity_level, density,
                super_assignment, c_min=c_min, rule=rule,
                enforce_global_balance=enforce_global_balance,
                psi_fn=psi_fn, super_psi_fn=super_psi_fn,
            )
        partition = cls._from_random_global(
            graph, epsilon, demand_target, capacity_level, density,
            c_min=c_min, rule=rule,
            enforce_global_balance=enforce_global_balance,
            psi_fn=psi_fn, super_psi_fn=super_psi_fn,
        )
        if init_super_partition:
            partition._init_super_partition_recursive(
                demand_target=demand_target,
                epsilon_super=epsilon_super if epsilon_super is not None else epsilon,
                density=density,
                super_psi_fn=super_psi_fn,
                c_min_super=c_min_super,
                c_max_super=c_max_super,
                min_districts_super=min_districts_super,
            )
        return partition

    def _init_super_partition_recursive(
        self,
        *,
        demand_target,
        epsilon_super,
        density,
        super_psi_fn: Optional[Callable] = None,
        c_min_super: int = 1,
        c_max_super: Optional[int] = None,
        min_districts_super: int = 1,
    ) -> None:
        """
        Replace the identity ``super_assignment`` with a real level-2
        partition produced by recursive partitioning of the supergraph
        (paper Section 5.1, "Initialization").

        On failure (e.g. supergraph too small to partition cleanly), the
        identity ``super_assignment`` is left in place and a warning is
        emitted. Callers should not rely on the operation succeeding;
        the resulting Partition is feasible either way.
        """
        import warnings

        total_teams = sum(self.teams.values())
        try:
            super_flip = capacitated_recursive_tree(
                graph=self.supergraph.copy(),
                n_teams=total_teams,
                demand_target=demand_target,
                epsilon=epsilon_super,
                capacity_level=(c_max_super
                                if c_max_super is not None
                                else self.capacity_level),
                density=density,
                c_min=c_min_super,
                supergraph=True,
                super_psi_fn=super_psi_fn,
                min_districts_super=min_districts_super,
            )
        except RuntimeError as exc:
            warnings.warn(
                f"init_super_partition: recursive supergraph partitioning "
                f"failed ({exc}); keeping identity super_assignment.",
                stacklevel=3,
            )
            return

        # super_flip.flips maps each level-1 district ID -> super ID.
        if super_flip.flips:
            self.super_assignment = dict(super_flip.flips)

    @classmethod
    def _from_random_global(
        cls, graph, epsilon, demand_target, capacity_level, density,
        c_min: int = 1, rule: str = "per_team",
        enforce_global_balance: bool = False,
        psi_fn: Optional[Callable] = None,
        super_psi_fn: Optional[Callable] = None,
    ) -> "Partition":
        total_pop = sum(graph.nodes[n]["demand"] for n in graph)
        n_teams = math.ceil(total_pop / demand_target)

        flip = capacitated_recursive_tree(
            graph=graph,
            n_teams=n_teams,
            demand_target=demand_target,
            epsilon=epsilon,
            capacity_level=capacity_level,
            density=density,
            c_min=c_min,
            rule=rule,
            enforce_global_balance=enforce_global_balance,
            psi_fn=psi_fn,
            super_psi_fn=super_psi_fn,
        )

        return cls(
            capacity_level=capacity_level,
            assignment=flip.flips,
            graph=graph,
            flip=flip,
        )

    @classmethod
    def _from_random_within_zones(
        cls, graph, epsilon, demand_target, capacity_level, density,
        super_assignment: Dict, c_min: int = 1, rule: str = "per_team",
        enforce_global_balance: bool = False,
        psi_fn: Optional[Callable] = None,
        super_psi_fn: Optional[Callable] = None,
    ) -> "Partition":
        """Partition each superdistrict (zone) independently and stitch."""
        # Validate: every node in the graph must have a superdistrict.
        missing = [n for n in graph.nodes if n not in super_assignment]
        if missing:
            raise ValueError(
                f"super_assignment is missing {len(missing)} graph node(s); "
                f"each base node must be assigned to a superdistrict. "
                f"First missing: {missing[:5]}"
            )

        # Group nodes by zone.
        zones = {}
        for node, zone_id in super_assignment.items():
            if node in graph.nodes:
                zones.setdefault(zone_id, set()).add(node)

        all_flips = {}
        all_team_flips = {}
        zone_to_l1_ids = {}
        next_id_offset = 0
        log_proposal_ratio = 0.0

        for zone_id, zone_nodes in zones.items():
            zone_subgraph = graph.subgraph(zone_nodes)
            zone_pop = sum(graph.nodes[n]["demand"] for n in zone_nodes)
            zone_teams = math.ceil(zone_pop / demand_target)
            if zone_teams < 1:
                raise ValueError(
                    f"Superdistrict {zone_id!r} has total demand {zone_pop} < "
                    f"demand_target {demand_target}; cannot allocate any team."
                )

            zone_flip = capacitated_recursive_tree(
                graph=zone_subgraph,
                n_teams=zone_teams,
                demand_target=demand_target,
                epsilon=epsilon,
                capacity_level=capacity_level,
                density=density,
                c_min=c_min,
                rule=rule,
                enforce_global_balance=enforce_global_balance,
                psi_fn=psi_fn,
                super_psi_fn=super_psi_fn,
            )
            log_proposal_ratio += zone_flip.log_proposal_ratio

            # Re-number this zone's level-1 IDs to avoid collisions across zones.
            id_remap = {
                old_id: i + next_id_offset
                for i, old_id in enumerate(sorted(zone_flip.new_ids))
            }
            next_id_offset += len(zone_flip.new_ids)

            for node, old_id in zone_flip.flips.items():
                all_flips[node] = id_remap[old_id]
            for old_id, n_teams in zone_flip.team_flips.items():
                all_team_flips[id_remap[old_id]] = n_teams
            zone_to_l1_ids[zone_id] = {id_remap[old_id] for old_id in zone_flip.new_ids}

        # Build the global Flip with renumbered IDs.
        merged_flip = Flip(
            flips=all_flips,
            team_flips=all_team_flips,
            new_ids=frozenset(all_team_flips.keys()),
            merged_ids=frozenset(),
            log_proposal_ratio=log_proposal_ratio,
        )

        partition = cls(
            capacity_level=capacity_level,
            assignment=all_flips,
            graph=graph,
            flip=merged_flip,
        )
        # Override the identity super_assignment with the zone grouping.
        zone_for_l1 = {}
        for zone_id, l1_ids in zone_to_l1_ids.items():
            for l1_id in l1_ids:
                zone_for_l1[l1_id] = zone_id
        partition.super_assignment = zone_for_l1
        return partition

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
        elif isinstance(graph, FrozenGraph):
            self.graph = graph
        elif isinstance(graph, nx.Graph):
            graph = Graph.from_networkx(graph)
            self.graph = FrozenGraph(graph)
        else:
            raise TypeError(f"Unsupported Graph object with type {type(graph)}")


        self.step = 1
        self.parent = None
        self.capacity_level = capacity_level
        
        self.flip = flip
        self.superflip = None
        self.flow = Flow.initial(flip)
        self.assignment = get_assignment(assignment, graph, flip.team_flips)
        # Initial level-2 partition: each level-1 district is its own
        # superdistrict (identity). Real groupings appear after the first
        # hierarchical_recom step.
        self.super_assignment = {pid: pid for pid in self.parts}

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
        self.super_assignment = self._build_super_assignment(parent, superflip, flip)

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

    @property
    def super_parts(self):
        """Maps superdistrict ID to the frozenset of level-1 district IDs in it."""
        groups = {}
        for l1_id, super_id in self.super_assignment.items():
            groups.setdefault(super_id, set()).add(l1_id)
        return {sid: frozenset(ls) for sid, ls in groups.items()}

    @property
    def super_teams(self):
        """Maps superdistrict ID to the total number of teams in it."""
        totals = {}
        for l1_id, super_id in self.super_assignment.items():
            totals[super_id] = totals.get(super_id, 0) + self.teams.get(l1_id, 0)
        return totals

    def _build_super_assignment(self, parent, superflip, flip):
        """
        Derive the new level-2 assignment after a hierarchical_recom step.

        The full resampled level-2 partition lives on ``superflip.flips``
        (level-1 ID -> superdistrict ID, computed over the parent's
        level-1 IDs). Three groups of level-1 IDs need handling:

        * **Unchanged** (parent IDs not in ``superflip.merged_ids``) keep
          the resampled super_id from ``superflip.flips``.
        * **Merged** (in ``superflip.merged_ids``) disappear in the new
          partition and are dropped.
        * **Newly created** (``flip.new_ids`` from the lower-level
          re-partition) all join the chosen superdistrict.

        For backwards compatibility with callers that still pass a
        ``Flip(merged_ids=merge)`` without populating ``flips``, this
        falls back to copying the parent's super_assignment, dropping
        merged IDs, and placing new IDs into the chosen superdistrict
        derived from any merged ID.
        """
        if superflip is None:
            return {pid: pid for pid in self.parts}

        merged = superflip.merged_ids or frozenset()

        if superflip.flips:
            chosen_super_id = (
                superflip.flips.get(next(iter(merged))) if merged else None
            )
            new_sa = {
                l1_id: super_id
                for l1_id, super_id in superflip.flips.items()
                if l1_id not in merged
            }
        else:
            # Legacy path: superflip carries only merged_ids. Inherit the
            # parent's level-2 partition and patch the chosen group.
            chosen_super_id = next(
                (parent.super_assignment[m] for m in merged
                 if m in parent.super_assignment),
                None,
            )
            new_sa = {
                l1_id: super_id
                for l1_id, super_id in parent.super_assignment.items()
                if l1_id not in merged
            }

        if chosen_super_id is not None:
            for new_l1_id in flip.new_ids:
                new_sa[new_l1_id] = chosen_super_id

        return new_sa


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


