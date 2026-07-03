"""
This module provides tools and algorithms for manipulating and analyzing graphs,
particularly focused on partitioning graphs based on demand data. It leverages the
NetworkX library to handle graph structures and implements various algorithms for graph
partitioning and tree traversal.

Key functionalities include:

- A spanning tree class that keeps accumulated weights for each node. Accumulation starts from leaves
  and each node keeps total weights of the subtrees beneath the node for searching a cut edge using breadth-first search.
- Random and uniform spanning tree generation for graph partitioning.
- Search functions for finding cut edges in a tree and supertree.
- Functions for finding balanced edge cuts in a demand graph

Dependencies:

- networkx: Used for graph data structure and algorithms.
- random: Provides random number generation for probabilistic approaches.
- typing: Used for type hints.

Last Updated: 8 October 2024
"""

import math
from collections import Counter, deque, namedtuple
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import networkx as nx
from networkx.algorithms import tree

# Diagnostic side-channel: when enabled, records (recursion_step,
# debt_value, tau_1) per iteration of capacitated_recursive_tree.
# Used by the per-team vs main-rule comparison harness to measure
# the empirical distribution of |δ|/τ during a chain run.
# This is debug instrumentation only; chain behaviour is unchanged.
DEBT_LOG_ENABLED = False
DEBT_LOG: list = []

# Per-call attempt counter for bipartition_tree. When enabled, every
# call appends a dict {attempts: int, success: bool, supergraph: bool}.
# Used to count rejections in the rule comparison harness.
ATTEMPT_LOG_ENABLED = False
ATTEMPT_LOG: list = []

from falcomchain.random import rng

from falcomchain.helper import save_pickle
from falcomchain.tree.errors import (
    BalanceError,
    BipartitionWarning,
    PopulationBalanceError,
    ReselectException,
)


# Not: write a spanning tree update function and use when generating a new spanning tree in bipartition_tree
#       we are passing the same parameters everytime
class SpanningTree:
    """
    A class representing a spanning tree with population and density information of its subtrees.

    :ivar graph: The underlying graph structure.
    :type graph: nx.Graph
    :ivar subsets: A dictionary mapping nodes to their subsets.
    :type subsets: Dict
    :ivar population: A dictionary mapping nodes to their populations.
    :type population: Dict
    :ivar total_demand: The total demand of the graph.
    :type total_demand: Union[int, float]
    :ivar ideal_demand: The ideal demand for each district.
    :type ideal_demand: float
    :ivar epsilon: The tolerance for demand deviation from the ideal demand within each district.
    :type epsilon: float
    :ivar preccessor: The predecessor
    :type preccessor: Dict
    :ivar successor:
    :type successor: Dict
    :ivar
    :type
    """

    __slots__ = (
        "graph",
        "root",
        "successors",
        "total_demand",
        "supertree",
        "tot_candidates",
        "candidate_nodes",
        "node_demands",
        "params",
    )

    def __init__(
        self,
        graph,
        params: "CutParams",
        supergraph: Optional[bool] = False,
    ) -> None:

        self.supertree = supergraph
        self.graph = graph
        self.params = params
        # Prefer non-leaf nodes for the root, but fall back to any node
        # (needed for small residual graphs during recursive partitioning)
        internal_nodes = [n for n in self.graph.nodes if self.graph.degree(n) > 1]
        if internal_nodes:
            self.root = rng.choice(internal_nodes)
        else:
            self.root = rng.choice(list(self.graph.nodes))

        # Record ORIGINAL per-node demand before accumulation overwrites the
        # "demand" field with subtree totals. Needed for the demand-weighted
        # 1-median geometric penalty η (paper Section 5.4).
        self.node_demands = {
            node: self.graph.nodes[node]["demand"] for node in self.graph.nodes
        }

        if self.supertree:
            self.candidate_nodes = frozenset()
            accumulation_columns = {"demand", "area", "n_teams"}
        else:
            # Record original candidate nodes BEFORE accumulation overwrites the field
            self.candidate_nodes = frozenset(
                node for node in self.graph.nodes
                if self.graph.nodes[node]["candidate"] == 1
            )
            self.tot_candidates = len(self.candidate_nodes)
            accumulation_columns = {"demand", "area", "candidate"}

        self.successors = self.find_successors()
        accumulate_tree(self, accumulation_columns)
        self.total_demand = self.graph.nodes[self.root]["demand"]

    # convenience shorthands so callers don't have to write h.params.epsilon etc.
    @property
    def ideal_demand(self):
        return self.params.ideal_demand

    @property
    def epsilon(self):
        return self.params.epsilon

    @property
    def capacity_level(self):
        return self.params.capacity_level

    @property
    def c_min(self):
        return self.params.c_min

    @property
    def n_teams(self):
        return self.params.n_teams

    @property
    def two_sided(self):
        return self.params.two_sided

    def find_successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self.graph, self.root)}

    def has_ideal_demand(self, assign_team, pop):
        if self.params.rule == "main":
            # Paper §5.3 main rule with capacity-scaled tolerance and
            # active debt correction. τ(c) = ε·c·d̄ matches the global-
            # balance gate at c=c_max. The window keeps width 2·τ(c) but
            # is *shifted* by 2δ in the corrective direction: when prior
            # extractions over-shot (δ > 0), U is pulled down by 2δ to
            # force the next district to undershoot; symmetrically for
            # δ < 0. This caps |δ| at τ_min/2 by construction.
            c = assign_team
            d_bar = self.ideal_demand
            tau = self.epsilon * c * d_bar
            debt = self.params.debt
            L = c * d_bar - tau + 2.0 * max(-debt, 0.0)
            U = c * d_bar + tau - 2.0 * max(debt, 0.0)
            return L <= pop <= U
        return abs(pop - assign_team * self.ideal_demand) <= self.ideal_demand * self.epsilon

    def complement_has_the_ideal_demand(self, assign_team, pop):
        # Same role as has_ideal_demand but for the complement T_{u'} taking
        # the same assign_team count (used in one_sided_cut).
        complement_demand = self.total_demand - pop
        if self.params.rule == "main":
            c = assign_team
            d_bar = self.ideal_demand
            tau = self.epsilon * c * d_bar
            debt = self.params.debt
            L = c * d_bar - tau + 2.0 * max(-debt, 0.0)
            U = c * d_bar + tau - 2.0 * max(debt, 0.0)
            return L <= complement_demand <= U
        return (
            abs(complement_demand - assign_team * self.ideal_demand)
            <= self.ideal_demand * self.epsilon
        )

    def complement_has_ideal_demand_too(self, assign_team, pop):
        # Two-sided check: the complement takes (n_teams - assign_team)
        # teams. Both halves of the cut must be balanced.
        complement_capacity = self.n_teams - assign_team
        complement_demand = self.total_demand - pop
        if self.params.rule == "main":
            c = complement_capacity
            d_bar = self.ideal_demand
            tau = self.epsilon * c * d_bar
            debt = self.params.debt
            L = c * d_bar - tau + 2.0 * max(-debt, 0.0)
            U = c * d_bar + tau - 2.0 * max(debt, 0.0)
            return L <= complement_demand <= U
        return (
            abs(complement_demand - complement_capacity * self.ideal_demand)
            <= self.ideal_demand * self.epsilon
        )

    def has_ideal_density(self, node):
        "Checks if the subtree beneath a node has an ideal density up to tolerance 'density'."
        return

    def has_facility(self, node):
        return self.graph.nodes[node]["candidate"] > 0

    def complement_has_facility(self, node):
        return (
            self.graph.nodes[node]["candidate"] < self.tot_candidates
            or node == self.root
        )

    def pop_remarkable_nodes(self):
        return {
            node
            for node, data in self.graph.nodes(data=True)
            if data["demand"] > 2 * self.ideal_demand / 3
        }

    def facility_remarkable_nodes(self):
        nodes = {
            n
            for n, attr in self.graph.nodes(data=True)
            if 0
            < attr["candidate"]
            < self.tot_candidates  # note: used for two_sided. < tot_candidates guarantees a candidate in the complement
        }
        nodes.add(self.root)
        return nodes

    def team_remarkable_nodes(self):
        nodes = {
            n
            for n, attr in self.graph.nodes(data=True)
            if 0
            < attr["n_teams"]
            < self.capacity_level  # note: used for two_sided. < tot_candidates guarantees a candidate in the complement
        }
        return

    def facility_remarkable_nodes_one_sided(self):
        nodes = {
            node for node, attr in self.graph.nodes(data=True) if attr["candidate"] > 0
        }
        return nodes

    def psi(self, node) -> float:
        """
        Candidate-awareness score for the subtree rooted at ``node``:

            psi(T_u) = phi(u) * exp(-gamma * r(T_u))

        where phi(u) is the facility indicator (number of candidates in subtree)
        and r(T_u) = min_{f in F_H ∩ T_u} e(f, T_u) is the demand radius —
        the minimum eccentricity over all facility candidates in T_u.

        When gamma = 0 this reduces to phi(u) (pure feasibility score).
        When travel_times is None, uses 1/phi as a proxy for r(T_u).
        """
        phi = self.graph.nodes[node]["candidate"]  # accumulated candidate count
        if phi == 0:
            return 0.0

        # Allow custom psi function
        if self.params.psi_fn is not None:
            r = self._demand_radius(node)
            return self.params.psi_fn(phi, self.params.gamma, r)

        gamma = self.params.gamma
        if gamma == 0.0:
            return float(phi)

        r = self._demand_radius(node)
        return phi * math.exp(-gamma * r)

    def _demand_radius(self, node) -> float:
        """
        Geometric penalty η¹(T_u): the demand-weighted MEAN distance from the
        subtree's demand-weighted 1-median to its nodes,

            η¹(T_u) = ( min_{f∈F¹∩T_u} Σ_{v∈T_u} d_v·dist(f,v) ) / Σ_{v∈T_u} d_v.

        This is the per-capita access cost of the best-located candidate — the
        median analogue of the old minimax covering radius — and is the
        quantity the disaggregated-median objective minimises (paper Section
        5.4). Normalising by subtree demand keeps η scale-free so it is
        comparable across cuts of different sizes inside exp(-γ·η).

        Falls back to 1/phi when travel_times is not available.
        """
        travel_times = self.params.travel_times
        if travel_times is None:
            phi = self.graph.nodes[node]["candidate"]
            return 1.0 / max(phi, 1)

        subtree_nodes = _part_nodes(self.successors, node)
        candidates_in_subtree = self.candidate_nodes & subtree_nodes

        if not candidates_in_subtree:
            return float("inf")

        total_demand = sum(self.node_demands.get(v, 1.0) for v in subtree_nodes)
        if total_demand <= 0:
            return float("inf")

        # min over candidates f of demand-weighted total distance; normalise.
        best_cost = float("inf")
        for f in candidates_in_subtree:
            cost = sum(
                self.node_demands.get(v, 1.0) * travel_times.get((f, v), float("inf"))
                for v in subtree_nodes
            )
            if cost < best_cost:
                best_cost = cost

        return best_cost / total_demand


def accumulate_tree(tree: SpanningTree, accumulation_columns):
    """
    Accumulates demand, area and facility attributes for the subtree under
    each node by traversing the graph using a depth-first search.
    return: None
    """
    accumulated = set()
    stack = deque([(tree.root)])

    while stack:
        node = stack.pop()
        children = tree.successors.get(node, [])
        if all(
            c in accumulated for c in children
        ):  # all children are processed, accumulate attributes from children to node
            for column in accumulation_columns:
                tree.graph.nodes[node][column] += sum(
                    tree.graph.nodes[c][column] for c in children
                )
            accumulated.add(node)
        else:
            stack.append(node)
            for c in children:
                if c not in accumulated:
                    stack.append(c)


def random_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph: The input graph to build the spanning tree from. Should be a Networkx Graph.
    :type graph: nx.Graph

    :returns: The minimum spanning tree represented as a Networkx Graph.
    :rtype: nx.Graph
    """
    for edge in graph.edges():
        weight = rng.random()
        graph.edges[edge]["random_weight"] = weight

    spanning_tree = tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
    return spanning_tree


def uniform_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph using Wilson's algorithm (loop-erased
    random walk).

    :param graph: Networkx Graph
    :type graph: nx.Graph

    :returns: A spanning tree of the graph chosen uniformly at random.
    :rtype: nx.Graph
    """
    nodes = list(graph.nodes)
    root = rng.choice(nodes)
    tree_nodes = {root}
    next_node = {root: None}

    for node in nodes:
        u = node
        while u not in tree_nodes:
            next_node[u] = rng.choice(list(graph.neighbors(u)))
            u = next_node[u]

        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    G.add_nodes_from(graph.nodes(data=True))

    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    return G


"""------------------------------------------------------------------------------------------------------------------------"""


def _part_nodes(successors, start):
    """
    Partitions the nodes of a graph into two sets.
    based on the start node and the successors of the graph.

    :param start: The start node.
    :type start: Any
    :param succ: The successors of the graph.
    :type succ: Dict

    :returns: A set of nodes for a particular district (only one side of the cut).
    :rtype: Set
    """

    nodes = set()
    queue = deque([start])
    while queue:
        next_node = queue.pop()
        if next_node not in nodes:
            nodes.add(next_node)
            if next_node in successors:
                for c in successors[next_node]:
                    if c not in nodes:
                        queue.append(c)
    return nodes


def compute_subtree_nodes(tree, succ, root) -> Dict:
    """
    Precompute subtree nodes for all nodes.
    Returns a dict: node -> set of nodes in the subtree rooted at node.
    """
    subtree_nodes = {}

    def dfs(node):
        nodes_set = {node}
        for child in succ.get(node, []):
            nodes_set.update(dfs(child))
        subtree_nodes[node] = nodes_set
        return nodes_set

    dfs(root)
    return subtree_nodes


"""  ------------------------------ Main Functions ------------------------------  """

Cut = namedtuple("Cut", "node subnodes assigned_teams demand psi")
"""
Represents one admissible cut of a spanning tree.

Fields:
  node          – root node of the cut subtree
  subnodes      – frozenset of nodes on one side of the cut
  assigned_teams – number of doctor-nurse teams for this side
  demand        – total demand of subnodes
  psi           – candidate-awareness score ψ(e) = φ(u) · exp(-γ · r(T_u))
"""


@dataclass(frozen=True)
class CutParams:
    """
    Cut parameters for a spanning tree bipartition.
    Separates algorithmic settings from the tree structure itself.

    :ivar ideal_demand: Target demand per team.
    :ivar epsilon: Allowed relative deviation from ideal_demand.
    :ivar capacity_level: Maximum teams per district.
    :ivar n_teams: Total teams to allocate across the graph.
    :ivar two_sided: If True, both sides of the cut must be balanced.
    :ivar gamma: Tuning parameter for psi score. 0 -> psi = phi (feasibility count only).
    :ivar travel_times: Dict (facility, node) -> travel time. None falls back to proxy.
    :ivar psi_fn: Optional custom psi scoring function(phi, gamma, radius) -> float.
    """

    ideal_demand: float
    epsilon: float
    capacity_level: int
    n_teams: int
    two_sided: bool = False
    gamma: float = 0.0
    travel_times: Optional[Dict] = None
    psi_fn: Optional[Any] = None  # Callable[[int, float, float], float]
    # Minimum capacity per district (paper c^ℓ_min). Districts with team
    # count below this are not admissible. Default 1 preserves the
    # original behavior. Setting c_min > 1 lets users opt into a stricter
    # algorithm with looser Assumption 6.1 (since the smallest valid
    # district has demand (1−ε)·c_min·d̄ rather than (1−ε)·d̄).
    c_min: int = 1
    # Level-2 (supergraph) cut score. Called as ``super_psi_fn(subnodes, teams)
    # -> float`` and used by ``find_superedge_cuts`` when set. When ``None``,
    # the supergraph cut score falls back to the team-count formula
    # (paper's γ=0 case). See ``hub_coherence_psi_factory`` in
    # ``falcomchain.markovchain.super_scoring`` for the paper's Eq. 27 default.
    super_psi_fn: Optional[Any] = None  # Callable[[frozenset, int], float]
    recorder: Optional[Any] = None  # Recorder instance for substep recording
    # Admissibility rule. "per_team" (default) preserves current
    # behaviour: clip per-team interval, derive (d^(r), ε_r), check
    # |pop − c·d^(r)| ≤ d^(r)·ε_r. "main" applies the paper §5.3 main
    # rule directly: check pop ∈ [L^(r)(c), U^(r)(c)] where the window
    # has half-width τ = epsilon·ideal_demand and centre c·ideal_demand,
    # with debt asymmetrically pinning one bound. Under "main",
    # ideal_demand is interpreted as d̄ (unclipped) and `debt` carries
    # δ^(r-1) from the recursion.
    rule: str = "per_team"
    debt: float = 0.0
    # Minimum number of level-1 districts per super-district (paper's
    # min-L1-per-L2). Enforced only at the supergraph level by
    # ``find_superedge_cuts``: a super-cut is admissible only if BOTH it
    # and its complement contain at least this many supergraph nodes
    # (= L1 districts). Default 1 preserves the legacy behaviour where a
    # single c¹≥c²_min district could form a super-district on its own.
    # Set to 2 to match the MIP's ``min_l1_per_l2 = 2``.
    min_districts_super: int = 1


@dataclass(frozen=True)
class Flip:
    flips: Dict[Any, Any] = field(default_factory=dict)
    team_flips: Dict[Any, Any] = field(default_factory=dict)
    new_ids: frozenset = field(default_factory=frozenset)
    merged_ids: frozenset = field(default_factory=frozenset)
    log_proposal_ratio: float = 0.0

    def add_merged_ids(self, new: FrozenSet) -> "Flip":
        return replace(self, merged_ids=new)


def two_sided_cut(h: SpanningTree, density_check) -> List[Cut]:
    cuts = []
    nodes = h.facility_remarkable_nodes()

    for node in nodes:
        pop = h.graph.nodes[node]["demand"]
        for assign_team in range(h.c_min, min(h.capacity_level + 1, h.n_teams + 1)):

            if h.has_ideal_demand(assign_team, pop):
                if node == h.root:
                    # Root cut: take the entire tree as one district.
                    # No complement district exists, so psi is just the subtree score.
                    psi_subtree = h.psi(node)
                    if psi_subtree > 0:
                        cuts.append(
                            Cut(
                                node=node,
                                subnodes=frozenset(_part_nodes(h.successors, node)),
                                assigned_teams=assign_team,
                                demand=pop,
                                psi=psi_subtree,
                            )
                        )
                elif (
                    h.complement_has_ideal_demand_too(assign_team, pop)
                    and h.n_teams - assign_team > 0
                ):
                    psi_subtree = h.psi(node)
                    complement_node = h.graph.nodes[node]["candidate"]
                    complement_phi = h.tot_candidates - complement_node
                    psi_complement = (
                        float(complement_phi) if h.params.gamma == 0.0
                        else complement_phi * math.exp(-h.params.gamma / max(complement_phi, 1))
                    )
                    psi_score = psi_subtree * psi_complement
                    if psi_score > 0:
                        cuts.append(
                            Cut(
                                node=node,
                                subnodes=frozenset(_part_nodes(h.successors, node)),
                                assigned_teams=assign_team,
                                demand=pop,
                                psi=psi_score,
                            )
                        )
    return cuts


def one_sided_cut(h: SpanningTree, density_check):
    cuts = []
    nodes = h.graph.nodes

    for node in nodes:
        pop = h.graph.nodes[node]["demand"]

        for assign_team in range(h.c_min, min(h.capacity_level + 1, h.n_teams + 1)):
            if h.has_ideal_demand(assign_team, pop) and h.has_facility(node):
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(_part_nodes(h.successors, node)),
                        assigned_teams=assign_team,
                        demand=pop,
                        psi=h.psi(node),
                    )
                )
            elif h.complement_has_the_ideal_demand(
                assign_team, pop
            ) and h.complement_has_facility(node):
                complement_phi = h.tot_candidates - h.graph.nodes[node]["candidate"]
                psi_complement = (
                    float(complement_phi) if h.params.gamma == 0.0
                    else complement_phi * math.exp(-h.params.gamma / max(complement_phi, 1))
                )
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(
                            set(nodes) - _part_nodes(h.successors, node)
                        ),
                        assigned_teams=assign_team,
                        demand=(h.total_demand - pop),
                        psi=psi_complement,
                    )
                )
    return cuts


def find_edge_cuts(h: SpanningTree, density_check: Optional[float] = None) -> List[Cut]:
    """
    This function takes a SpanningTree object as input and returns a list of balanced edge cuts.
    A balanced edge cut is defined as a cut that divides the graph into two subsets, such that
    the demand of each subset is close to the ideal demand defined by the SpanningTree object.

    :param h: The SpanningTree object representing the graph.
    :param add_root: If set to True, an artifical node is connected to root and edge is considered as a possible cut.

    :returns: A list of balanced edge cuts.
    """
    # print("--------------iteration starts")
    # print("remaining demand", h.total_demand)

    if h.two_sided == True:
        cuts = two_sided_cut(h, density_check)
    else:
        cuts = one_sided_cut(h, density_check)

    # print("length of cuts", len(cuts))
    # print("ideal demand", h.ideal_demand)
    # print("epsilon", h.epsilon)
    # print("root demand", h.graph.nodes[h.root][h.demand_col])
    # print("root facility", h.graph.nodes[h.root][h.facility_col])
    # print("--------------iteration ends")

    return cuts


def find_superedge_cuts(
    h: SpanningTree,
    density_check=None,
) -> List[Cut]:
    """
    Find admissible cuts on the spanning tree of a supergraph.

    Admissibility uses the team-count balance condition (paper Eq. 13/14
    at level 2). The cut score ψ²(T_u) is computed by ``h.params.super_psi_fn``
    when set — the paper's hub-coherence penalty (Eq. 27) plugged in via
    :func:`falcomchain.markovchain.super_scoring.hub_coherence_psi_factory`.
    When ``super_psi_fn`` is ``None``, falls back to ϕ²(T_u) = teams (paper's
    γ=0 case).

    :param h: The SpanningTree object representing the supergraph.
    :returns: A list of admissible cuts (each with positive ψ²).
    """
    cuts = []
    nodes = h.graph.nodes
    super_psi_fn = h.params.super_psi_fn

    def _phi(subnodes, teams):
        """ϕ²(T_u) = teams (paper Eq. 21 with fac²=1) when no custom scorer."""
        return float(teams)

    # At the supergraph level, c_min is c²_min (= 2·c¹_min by convention,
    # verified in hierarchical_recom). The lower bound applies to BOTH the
    # leaf side (`teams`) and, in the one-sided branch, to its complement.
    c_min = h.c_min
    # Minimum L1 districts per super-district. At the supergraph level each
    # tree node IS one L1 district, so the district count of a super-cut is
    # the number of supergraph nodes in it.
    min_d = h.params.min_districts_super
    n_super_nodes = len(nodes)

    for node in nodes:
        teams = nodes[node]["n_teams"]
        pop = nodes[node]["demand"]

        if h.two_sided:
            if (
                teams >= c_min
                and abs(pop - teams * h.ideal_demand) <= h.ideal_demand * teams * h.epsilon
            ):
                subtree_set = _part_nodes(h.successors, node)
                leaf_count = len(subtree_set)
                complement_count = n_super_nodes - leaf_count
                # Leaf must have >= min_d districts. If not the root cut
                # (which takes the whole tree as one super), the complement
                # must too.
                if leaf_count < min_d:
                    continue
                if node != h.root and complement_count < min_d:
                    continue
                if (
                    node == h.root
                    or abs((h.total_demand - pop) - (h.n_teams - teams) * h.ideal_demand)
                    <= h.ideal_demand * (h.n_teams - teams) * h.epsilon
                ):
                    subnodes = frozenset(subtree_set)
                    psi = (
                        super_psi_fn(subnodes, teams)
                        if super_psi_fn is not None
                        else _phi(subnodes, teams)
                    )
                    cuts.append(
                        Cut(
                            node=node,
                            subnodes=subnodes,
                            assigned_teams=teams,
                            demand=pop,
                            psi=psi,
                        )
                    )
        else:  # one sided
            subtree_set = _part_nodes(h.successors, node)
            leaf_count = len(subtree_set)
            complement_count = n_super_nodes - leaf_count
            if (c_min <= teams <= h.capacity_level) and abs(
                pop - teams * h.ideal_demand
            ) <= h.ideal_demand * teams * h.epsilon and leaf_count >= min_d:
                subnodes = frozenset(subtree_set)
                psi = (
                    super_psi_fn(subnodes, teams)
                    if super_psi_fn is not None
                    else _phi(subnodes, teams)
                )
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=subnodes,
                        assigned_teams=teams,
                        demand=pop,
                        psi=psi,
                    )
                )
            else:
                # Complement-side branch: this node's subtree is not admissible
                # as the leaf, but the COMPLEMENT (the "rest" of the tree) might
                # be. Demand and team-count constraints must use the complement's
                # totals, not the leaf-side `teams` / `pop`.
                complement_teams = h.n_teams - teams
                complement_pop = h.total_demand - pop
                if (
                    c_min <= complement_teams <= h.capacity_level
                    and abs(
                        complement_pop - complement_teams * h.ideal_demand
                    ) <= h.ideal_demand * complement_teams * h.epsilon
                    and complement_count >= min_d
                ):
                    subnodes = frozenset(
                        set(nodes) - subtree_set
                    )
                    psi = (
                        super_psi_fn(subnodes, complement_teams)
                        if super_psi_fn is not None
                        else _phi(subnodes, complement_teams)
                    )
                    cuts.append(
                        Cut(
                            node=node,
                            subnodes=subnodes,
                            assigned_teams=complement_teams,
                            demand=complement_pop,
                            psi=psi,
                        )
                    )
    return cuts


def bipartition_tree(
    graph: nx.Graph,
    demand_target: Union[int, float],
    epsilon: float,
    capacity_level: int,
    n_teams: int,
    two_sided: bool,
    supergraph: bool,
    iteration: int = 0,
    density: Optional[float] = None,
    max_attempts=5000,
    allow_pair_reselection: bool = False,
    initial=False,
    tree_sampler=None,
    gamma: float = 0.0,
    travel_times=None,
    psi_fn=None,
    super_psi_fn=None,
    recorder=None,
    c_min: int = 1,
    rule: str = "per_team",
    debt: float = 0.0,
    enforce_global_balance: bool = False,
    tau_global: Optional[float] = None,
    d_bar_orig: Optional[float] = None,
    min_districts_super: int = 1,
) -> Cut:
    """
    Finds a balanced 2-partition of a graph by drawing a spanning tree and
    finding an edge to cut that leaves at most an epsilon imbalance between
    the demands of the parts. If no valid cut exists, a new tree is drawn.

    :param graph: The graph to partition.
    :param demand_target: The target demand for the returned subset of nodes.
    :param epsilon: The allowable deviation from ``demand_target``.
    :param capacity_level: Maximum teams per district.
    :param n_teams: Total teams for the subgraph.
    :param two_sided: If True, both sides of the cut must be balanced.
    :param supergraph: If True, use supergraph admissibility conditions.
    :param tree_sampler: Callable(nx.Graph) -> nx.Graph that produces a spanning tree.
        Defaults to ``uniform_spanning_tree`` (Wilson's algorithm).
    :param gamma: Candidate-awareness tuning parameter (>= 0). Default 0 (uniform).
    :param travel_times: Dict (facility, node) -> travel time for psi computation.
    :param psi_fn: Optional level-1 scoring function ``(phi, gamma, radius) -> float``.
    :param super_psi_fn: Optional level-2 scoring function
        ``(subnodes, teams) -> float`` used by ``find_superedge_cuts`` when
        ``supergraph=True``. Defaults to ϕ²(T_u) = teams (paper γ=0 case).
        See :mod:`falcomchain.markovchain.super_scoring` for the paper's
        Eq. 27 hub-coherence factory.
    :param max_attempts: Maximum spanning tree samples before giving up.
    :param allow_pair_reselection: If True, raise ReselectException instead of RuntimeError.

    :returns: A (Cut, log_cut_ratio) tuple.
    :raises RuntimeError: If no valid cut found after ``max_attempts``.
    """
    if tree_sampler is None:
        tree_sampler = uniform_spanning_tree

    for _attempt_idx in range(max_attempts):

        spanning_tree = tree_sampler(graph)

        h = SpanningTree(
            graph=spanning_tree,
            params=CutParams(
                ideal_demand=demand_target,
                epsilon=epsilon,
                capacity_level=capacity_level,
                n_teams=n_teams,
                two_sided=two_sided,
                gamma=gamma,
                travel_times=travel_times,
                psi_fn=psi_fn,
                super_psi_fn=super_psi_fn,
                recorder=recorder,
                c_min=c_min,
                rule=rule,
                debt=debt,
                min_districts_super=min_districts_super,
            ),
            supergraph=supergraph,
        )

        if h.supertree == False:
            possible_cuts = find_edge_cuts(h)
        else:
            possible_cuts = find_superedge_cuts(h)

        possible_cuts = [c for c in possible_cuts if c.psi > 0]
        if (
            enforce_global_balance
            and possible_cuts
            and tau_global is not None
            and d_bar_orig is not None
        ):
            # Reject cuts whose demand falls outside the paper's global
            # balance window I(c) = [c·d̄_orig − τ_global, c·d̄_orig + τ_global]
            # with τ_global = ε_orig · c_max · d̄_orig (paper convention).
            # `tau_global` and `d_bar_orig` are supplied by the caller using
            # the ORIGINAL ε and d̄, NOT the per-team-clipped values that
            # bipartition_tree itself sees as `demand_target` / `epsilon`.
            # Under the main rule, [L,U] ⊆ I(c) so this filter is a no-op.
            possible_cuts = [
                c_ for c_ in possible_cuts
                if abs(c_.demand - c_.assigned_teams * d_bar_orig) <= tau_global
            ]
        if possible_cuts:
            total_psi = sum(c.psi for c in possible_cuts)
            weights = [c.psi / total_psi for c in possible_cuts]
            chosen = rng.choices(possible_cuts, weights=weights, k=1)[0]
            log_cut_ratio = math.log(chosen.psi) - math.log(total_psi)

            # Record substep for animation
            rec = h.params.recorder
            if rec is not None:
                rec.record_tree_cut(
                    tree_edges=list(spanning_tree.edges()),
                    root=h.root,
                    cut_node=chosen.node,
                    psi_chosen=chosen.psi,
                    psi_total=total_psi,
                    n_cuts=len(possible_cuts),
                    spanning_tree_obj=h,
                    extracted_nodes=chosen.subnodes,
                )

            if ATTEMPT_LOG_ENABLED:
                ATTEMPT_LOG.append({
                    "attempts": _attempt_idx + 1,
                    "success": True,
                    "supergraph": bool(supergraph),
                    "rule": rule,
                    "capacity": int(chosen.assigned_teams),
                    "demand": float(chosen.demand),
                })
            return chosen, log_cut_ratio

    if ATTEMPT_LOG_ENABLED:
        ATTEMPT_LOG.append({
            "attempts": max_attempts,
            "success": False,
            "supergraph": bool(supergraph),
            "rule": rule,
        })

    if allow_pair_reselection:
        raise ReselectException(
            f"Failed to find a balanced cut after {max_attempts} attempts.\n"
            f"Selecting a new district pair."
        )

    raise RuntimeError(
        f"Could not find a possible cut after {max_attempts} attempts. Supergraph = {h.supertree}."
    )


def determine_district_id(ids, max_id, assignments, district_nodes):
    """
     Assigns district id for a set of newly selected nodes in the intermidate step.
     We first consider assigning the district id that was the district id of the
     most of the nodes in district_nodes. If this id is already choosen, and ids still
     have an id to use, we select a random id from ids. Otherwise, we use max_id.

    Args:
        ids (list): a set of numbers from 1 to n, where n is big enough to assign an id to each newly created district in intial partition
        max_id (int):
        assignments (dict): _description_
        district_nodes (set): _description_
    """
    if len(ids) > 0:
        remarkable_district_nodes = {
            node for node in district_nodes if assignments[node] in ids
        }

        if len(remarkable_district_nodes) > 0:
            assignment_counts = Counter(
                [assignments[node] for node in remarkable_district_nodes]
            )
            district = max(assignment_counts, key=assignment_counts.get)
            ids.remove(district)
        else:
            district = rng.choice(list(ids))
            ids.remove(district)
    else:
        district = max_id + 1
        max_id = max_id + 1

    return district, ids, max_id


def capacitated_recursive_tree(
    graph: nx.Graph,
    n_teams: int,
    demand_target: int,
    epsilon: float,
    capacity_level: int,
    density=None,
    supergraph=False,
    assignments=None,
    merged_ids=None,
    max_id=0,
    iteration=0,
    recorder=None,
    super_psi_fn=None,
    c_min: int = 1,
    max_attempts: int = 5000,
    gamma: float = 0.0,
    travel_times=None,
    psi_fn=None,
    rule: str = "per_team",
    enforce_global_balance: bool = False,
    min_districts_super: int = 1,
) -> Flip:
    """
     Recursively partitions a graph into balanced districts using bipartition_tree.

    :param graph: The graph to partition into ``len(parts)`` :math:`epsilon`-balanced parts.
    :param filtered_parts:
    :param n_parts:
    :param n_teams: Total number of doctor-nurse teams for all facilities.
    :param demand_target: Target demand for each part of the partition.
    :param column_names:
    :param epsilon: How far (as a percentage of ``demand_target``) from ``demand_target`` the parts of the partition can be.
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, If it is 1, n_teams many districts are created.
    :param density: Defaluts to None.
    :param ids: set of ids whose districts are merged
    :param assignments: Old assignments for the nodes of ``graph``.
    :param max_id: maximum district id that has been used before
    :returns: New assignments for the nodes of ``graph``.
    :rtype: dict
    """

    current_flips = {}  # maps nodes to their districts
    current_team_flips = {}  # maps districts to their number of teams
    current_new_ids = set()
    log_proposal_ratio = 0.0  # accumulated log(ψ_chosen / Σψ) over all cuts
    ids = merged_ids
    remaining_nodes = set(graph.nodes())
    remaining_teams = n_teams
    debt = 0
    hired_teams = 1

    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the demand constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # demand within +/-``epsilon`` of the target demand.
    # For instance, if district n's demand exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's demand should be between
    # 98% of the target demand and the target demand.
    # "Change  this later"
    # Capacity level update: We multiply min_demand and max_demand by capacity level of a
    # district to set its demand target correctly. This enlarges error bounds
    # for districts with high demand densities.

    # print(f"------ recursive function starts.")
    # print(f"num of teams {n_teams}.")
    # print(print(f"total demand {sum(graph.nodes[node]["demand"] for node in remaining_nodes)}."))
    min_demand = demand_target * (1 - epsilon)
    max_demand = demand_target * (1 + epsilon)
    check_demand = lambda x: min_demand <= x <= max_demand

    # new_epsilon = epsilon
    # new_demand_target = demand_target
    while remaining_teams > 0:  # better to take len(remaining_nodes) > 0

        two_sided = remaining_teams <= capacity_level

        if supergraph:
            # At the supergraph level, super-node demands are pinned at
            # ≈ c¹·w (one super-node = one L1 district carrying its
            # team-times-w workload). Debt adjustment was designed for
            # the base graph where individual node demands vary widely
            # and accumulated drift needs correcting. On the supergraph
            # the debt-adjusted window slides off the super-node scale
            # and admissible cuts disappear after the first few peels —
            # the recursion exhausts max_attempts without ever finding
            # a feasible cut. Skip the debt term here and use the fixed
            # window throughout.
            new_demand_target = demand_target
            new_epsilon = epsilon
        elif rule == "main":
            # Paper §5.3 main rule: pass (d̄, ε, debt) through to
            # bipartition_tree; admissibility uses [L^(r)(c), U^(r)(c)]
            # with half-width τ = ε·d̄ and asymmetric debt shift.
            new_demand_target = demand_target
            new_epsilon = epsilon
            if DEBT_LOG_ENABLED:
                DEBT_LOG.append({
                    "step": iteration,
                    "recursion": len(current_flips),
                    "debt": float(debt),
                    "tau_1": float(epsilon * demand_target),
                    "demand_target": float(demand_target),
                    "supergraph": bool(supergraph),
                    "rule": "main",
                })
        else:
            # Per-team rule (legacy, inherited from GerryChain's
            # recursive_tree_part). Clip per-team interval, derive
            # (d^(r), ε^(r)).
            min_demand = max(
                demand_target * (1 - epsilon),
                demand_target * (1 - epsilon) - debt,
            )
            max_demand = min(
                demand_target * (1 + epsilon),
                demand_target * (1 + epsilon) - debt,
            )
            new_demand_target = (min_demand + max_demand) / 2
            new_epsilon = (max_demand - min_demand) / (2 * new_demand_target)

            if DEBT_LOG_ENABLED:
                DEBT_LOG.append({
                    "step": iteration,
                    "recursion": len(current_flips),
                    "debt": float(debt),
                    "tau_1": float(epsilon * demand_target),
                    "demand_target": float(demand_target),
                    "supergraph": bool(supergraph),
                    "rule": "per_team",
                })

        # else:
        #    new_demand_target=1500
        #    new_epsilon=0.1
        # print("min demand, max demand:", min_demand, max_demand)
        # print("new demand target:", new_demand_target)
        # print("new epsilon:", new_epsilon)

        try:
            cut_object, log_cut_ratio = bipartition_tree(
                graph.subgraph(remaining_nodes),
                demand_target=new_demand_target,
                capacity_level=capacity_level,
                n_teams=remaining_teams,
                epsilon=new_epsilon,
                two_sided=two_sided,
                supergraph=supergraph,
                density=density,
                iteration=iteration,
                recorder=recorder,
                super_psi_fn=super_psi_fn,
                c_min=c_min,
                max_attempts=max_attempts,
                gamma=gamma,
                travel_times=travel_times,
                psi_fn=psi_fn,
                rule=rule if not supergraph else "per_team",
                debt=debt if not supergraph else 0.0,
                enforce_global_balance=enforce_global_balance and not supergraph,
                tau_global=(epsilon * capacity_level * demand_target)
                            if (enforce_global_balance and not supergraph) else None,
                d_bar_orig=demand_target
                            if (enforce_global_balance and not supergraph) else None,
                min_districts_super=min_districts_super if supergraph else 1,
            )

        except Exception:
            raise

        log_proposal_ratio += log_cut_ratio
        hired_teams = cut_object.assigned_teams
        # print("hired teams", hired_teams)
        # print("district demand:", cut_object.demand)
        district_nodes = cut_object.subnodes
        pop = cut_object.demand

        if not check_demand(pop / hired_teams):
            raise PopulationBalanceError()

        # determine district id
        if assignments == None:  # initial partitioning
            district_id = max_id + 1
            max_id += 1
        else:
            district_id, ids, max_id = determine_district_id(
                ids,
                max_id,
                assignments,
                district_nodes,  # we need a better function for this. (look also for remaining nodes)
            )
            assignments = {
                key: value
                for key, value in assignments.items()
                if key not in district_nodes
            }

        # assign number of hired teams to the district
        current_team_flips[district_id] = hired_teams

        # updates for the next iteration
        debt += pop - demand_target * hired_teams
        remaining_teams -= hired_teams

        remaining_nodes -= district_nodes
        current_flips.update({node: district_id for node in district_nodes})

        current_new_ids.add(district_id)

        iteration += 1

    # print("------ recursive function ends sucessfully.")
    
    return Flip(
        flips=current_flips,
        team_flips=current_team_flips,
        new_ids=frozenset(current_new_ids),
        log_proposal_ratio=log_proposal_ratio,
    )
