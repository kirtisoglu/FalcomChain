"""
Verification and repair of the facility density assumption (Assumption 6.1).

The FalCom chain is irreducible only when every connected component of
the facility-free subgraph ``G[V \\ F]`` has total demand strictly
less than ``(1 - epsilon) * demand_target``. If this fails, the chain
can get trapped: a feasible district can be formed without containing
any candidate facility, blocking valid proposals.

This module provides:

- :func:`check_facility_density` — verify the assumption (O(|V| + |E|)).
- :func:`repair_facility_density` — add artificial candidates until the
  assumption holds.

See Section 6 of the FalCom paper for the theoretical motivation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class FeasibilityReport:
    """
    Result of :func:`check_facility_density`.

    :ivar passes: True iff Assumption 6.1 holds.
    :ivar threshold: The demand bound ``(1 - epsilon) * demand_target``.
    :ivar violating_components: List of node sets, one per facility-free
        component whose demand is at or above the threshold.
    :ivar component_demands: Demand of each component in
        ``violating_components`` (same order).
    :ivar worst_demand: Maximum component demand observed (0.0 if no
        violating components).
    """

    passes: bool
    threshold: float
    violating_components: List[Set] = field(default_factory=list)
    component_demands: List[float] = field(default_factory=list)
    worst_demand: float = 0.0

    def __repr__(self):
        if self.passes:
            return f"<FeasibilityReport passes (threshold={self.threshold:.2f})>"
        return (
            f"<FeasibilityReport FAILS: {len(self.violating_components)} "
            f"violating components, worst demand={self.worst_demand:.2f}, "
            f"threshold={self.threshold:.2f}>"
        )


def check_facility_density(
    graph,
    demand_target: float,
    epsilon: float = 0.1,
    candidate_attr: str = "candidate",
    demand_attr: str = "demand",
    c_min: int = 1,
) -> FeasibilityReport:
    """
    Verify Assumption 6.1 (facility density) on ``graph``.

    Generalised threshold ``(1 − ε) · c_min · demand_target``. With
    ``c_min = 1`` (default) this matches the paper's stated form. With
    ``c_min > 1`` the threshold scales linearly because the chain will
    never propose a district with fewer than ``c_min`` teams; the
    smallest valid district has demand ``(1 − ε) · c_min · demand_target``.

    Runs a single BFS over the facility-free subgraph in O(|V| + |E|).

    :param graph: A networkx Graph (or compatible) with ``demand`` and
        ``candidate`` node attributes.
    :param demand_target: Per-team workload (paper's ``w``).
    :param epsilon: Allowed relative deviation from the demand target.
    :param candidate_attr: Node attribute used to identify candidates
        (truthy = candidate).
    :param demand_attr: Node attribute holding per-node demand.
    :param c_min: Minimum capacity per district (paper's ``c^ℓ_min``).
        Default 1. Larger values relax Assumption 6.1.

    :returns: A :class:`FeasibilityReport` describing pass/fail and any
        violating components.
    """
    threshold = (1.0 - epsilon) * c_min * demand_target
    facility_free = {
        n for n in graph.nodes if not graph.nodes[n].get(candidate_attr, 0)
    }

    visited = set()
    violating = []
    demands = []
    worst = 0.0

    for start in facility_free:
        if start in visited:
            continue
        component = _bfs_component(graph, start, facility_free, visited)
        d = sum(graph.nodes[n].get(demand_attr, 0.0) for n in component)
        if d > worst:
            worst = d
        if d >= threshold:
            violating.append(component)
            demands.append(d)

    return FeasibilityReport(
        passes=(len(violating) == 0),
        threshold=threshold,
        violating_components=violating,
        component_demands=demands,
        worst_demand=worst,
    )


def feasibility_violation(
    graph,
    demand_target: float,
    epsilon: float = 0.1,
    candidate_attr: str = "candidate",
    demand_attr: str = "demand",
    c_min: int = 1,
) -> float:
    """
    Quantify how badly Assumption 6.1 is violated on ``graph``.

    Returns the *relative* worst-component-demand overshoot:

    ``violation = max(0, (worst_demand - threshold) / threshold)``

    where ``threshold = (1 − ε) · c_min · demand_target``.

    A return value of 0 means 6.1 holds. A return value of 0.5 means
    the worst facility-free component has 50 % more demand than the
    threshold allows.

    This is the building block for soft-constrained chain runs: an
    acceptance rule like
    ``soft_constraint_accept(violation_fn=feasibility_violation, beta_feas=...)``
    lets the chain explore states whose underlying graph slightly
    violates 6.1 with proportionally lower stationary mass.

    :param graph: networkx Graph with ``demand`` and ``candidate`` attrs.
    :param demand_target: Per-team workload (paper's ``w``).
    :param epsilon: Allowed relative deviation.
    :param candidate_attr: Node attribute used to identify candidates.
    :param demand_attr: Node attribute holding per-node demand.
    :param c_min: Minimum capacity per district.
    :returns: ``0.0`` if the graph satisfies Assumption 6.1, else the
        relative overshoot of the worst facility-free component.
    """
    report = check_facility_density(
        graph,
        demand_target=demand_target,
        epsilon=epsilon,
        candidate_attr=candidate_attr,
        demand_attr=demand_attr,
        c_min=c_min,
    )
    if report.passes or report.threshold == 0:
        return 0.0
    return max(0.0, (report.worst_demand - report.threshold) / report.threshold)


def repair_facility_density(
    graph,
    demand_target: float,
    epsilon: float = 0.1,
    strategy: str = "weighted_center",
    candidate_attr: str = "candidate",
    demand_attr: str = "demand",
    artificial_attr: str = "candidate_artificial",
    max_iterations: Optional[int] = None,
    c_min: int = 1,
) -> Set:
    """
    Add artificial candidates to ``graph`` until Assumption 6.1 holds.

    Modifies ``graph`` in place: each added node gets
    ``graph.nodes[n][candidate_attr] = 1`` and
    ``graph.nodes[n][artificial_attr] = 1``. Pre-existing candidates
    are not modified.

    :param graph: Graph to repair (modified in place).
    :param demand_target: Per-team workload (paper's ``w``).
    :param epsilon: Allowed relative deviation.
    :param strategy: Placement rule for the new candidate(s) within
        each violating component. Single-vertex strategies:

        - ``"highest_demand"`` — node with maximum demand (paper's procedure).
          O(|V|) per iteration; fastest, but on planar graphs the
          peripheral pick rarely fragments the component, so far more
          candidates are needed than the structural lower bound.
        - ``"center"`` — exact unweighted graph 1-center (minimax
          distance). O(|V|·(|V|+|E|)) per iteration; does not scale
          beyond a few thousand-node components.
        - ``"weighted_center"`` — exact demand-weighted 1-center.
          Same complexity as ``"center"``. Default.
        - ``"fast_center"`` — approximate graph 1-center via the
          double-BFS diameter trick (Roditty-Williams 2013):
          three BFS calls per iteration, O(|V|+|E|). Exact on trees,
          near-optimal on planar graphs.

        Multi-vertex (separator) strategies — add a *set* of nodes per
        iteration and tend to produce far fewer total candidates on
        planar graphs of average degree ≥ 4:

        - ``"balanced_separator"`` — approximate balanced vertex
          separator via BFS-based bisection (2-BFS for diameter
          endpoints, partition by reach distance, mark the border).
          Adds O(√|V|) candidates per iteration. **Note:** this
          balances by graph distance, not demand — on graphs with
          heterogeneous vertex demand it underperforms
          ``"fast_center"``. Use ``"metis_separator"`` for
          demand-balanced cuts.
        - ``"metis_separator"`` — demand-balanced vertex separator
          via METIS (multilevel graph partitioner). Requires the
          ``pymetis`` package (``pip install pymetis``). The cut is
          demand-balanced via ``vweights`` and tends to produce the
          smallest scaffolds on heterogeneous-demand graphs (LAS
          LSOAs, real-data planar graphs).
    :param candidate_attr: Node attribute used to mark candidates.
    :param demand_attr: Node attribute holding demand.
    :param artificial_attr: Node attribute used to flag added candidates.
    :param max_iterations: Optional cap on outer iterations. Defaults to
        ``len(graph.nodes) + 1`` (no node can be added twice).
    :param c_min: Minimum capacity per district (paper's ``c^ℓ_min``).
        Default 1. With ``c_min > 1`` the threshold scales linearly,
        and far fewer artificial candidates are needed.

    :returns: The set of nodes added as artificial candidates.

    :raises ValueError: If ``strategy`` is not recognized, or if
        ``max_iterations`` is exceeded (indicates a bug, not a real
        infeasibility — the procedure is guaranteed to terminate).
    """
    if strategy not in (
        "highest_demand",
        "center",
        "weighted_center",
        "fast_center",
        "balanced_separator",
        "metis_separator",
    ):
        raise ValueError(
            f"Unknown strategy {strategy!r}. Use 'highest_demand', "
            "'center', 'weighted_center', 'fast_center', "
            "'balanced_separator', or 'metis_separator'."
        )

    if max_iterations is None:
        max_iterations = len(graph.nodes) + 1

    added: Set = set()
    for _ in range(max_iterations):
        report = check_facility_density(
            graph,
            demand_target=demand_target,
            epsilon=epsilon,
            candidate_attr=candidate_attr,
            demand_attr=demand_attr,
            c_min=c_min,
        )
        if report.passes:
            return added

        for component in report.violating_components:
            new_candidates = _pick_candidates(
                graph, component, strategy, demand_attr
            )
            for n in new_candidates:
                graph.nodes[n][candidate_attr] = 1
                graph.nodes[n][artificial_attr] = 1
                added.add(n)

    raise ValueError(
        f"repair_facility_density did not converge after {max_iterations} "
        "iterations. This indicates a bug; please report it."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _bfs_component(graph, start, allowed, visited):
    """BFS over ``allowed`` starting from ``start``, marking ``visited``."""
    component = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        n = queue.popleft()
        component.add(n)
        for nbr in graph.neighbors(n):
            if nbr in allowed and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return component


def _pick_candidates(graph, component, strategy, demand_attr) -> Set:
    """Pick one or more nodes from ``component`` to promote.

    Returns a set: single-vertex strategies return a singleton; the
    ``balanced_separator`` strategy returns the full separator set
    (typically O(√|component|) nodes on planar graphs).
    """
    if strategy == "highest_demand":
        return {
            max(component, key=lambda n: graph.nodes[n].get(demand_attr, 0.0))
        }

    if strategy == "fast_center":
        return {_approximate_center(graph, component)}

    if strategy == "balanced_separator":
        return _approximate_balanced_separator(graph, component)

    if strategy == "metis_separator":
        return _metis_balanced_separator(graph, component, demand_attr)

    distances = {n: _bfs_distances(graph, n, component) for n in component}

    if strategy == "center":
        return {min(component, key=lambda n: max(distances[n].values()))}

    # weighted_center
    def weighted_eccentricity(node):
        d_map = distances[node]
        return max(
            graph.nodes[u].get(demand_attr, 0.0) * d_map[u] for u in component
        )

    return {min(component, key=weighted_eccentricity)}


def _bfs_distances(graph, source, allowed):
    """Unweighted graph distances from ``source`` within ``allowed``."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        n = queue.popleft()
        for nbr in graph.neighbors(n):
            if nbr in allowed and nbr not in dist:
                dist[nbr] = dist[n] + 1
                queue.append(nbr)
    return dist


def _bfs_with_predecessors(graph, source, allowed):
    """BFS within ``allowed``; return (dist, pred) maps."""
    dist = {source: 0}
    pred = {source: None}
    queue = deque([source])
    while queue:
        n = queue.popleft()
        for nbr in graph.neighbors(n):
            if nbr in allowed and nbr not in dist:
                dist[nbr] = dist[n] + 1
                pred[nbr] = n
                queue.append(nbr)
    return dist, pred


def _metis_balanced_separator(graph, component, demand_attr) -> Set:
    """
    Demand-balanced vertex separator of ``component`` via METIS.

    Uses ``pymetis.part_graph`` with ``vweights`` set to per-node
    demand to produce a 2-way partition with balanced demand on each
    side. The separator is the boundary of the smaller side: nodes
    on side 0 with at least one neighbour on side 1.

    On planar graphs with heterogeneous demand this typically gives
    significantly smaller separators than the unweighted BFS-based
    ``balanced_separator`` strategy, because the cut respects demand
    rather than graph distance.

    Requires ``pymetis``. ``ImportError`` if not installed.
    """
    try:
        import numpy as np
        import pymetis
    except ImportError as e:
        raise ImportError(
            "metis_separator strategy requires pymetis. "
            "Install with: pip install pymetis"
        ) from e

    if len(component) <= 1:
        return set(component)

    nodes = list(component)
    n2i = {n: i for i, n in enumerate(nodes)}

    adj = [
        np.array(
            [n2i[m] for m in graph.neighbors(n) if m in component],
            dtype=np.int32,
        )
        for n in nodes
    ]

    # METIS requires integer vertex weights ≥ 1.
    vw = np.array(
        [
            max(1, int(round(graph.nodes[n].get(demand_attr, 1.0))))
            for n in nodes
        ],
        dtype=np.int32,
    )

    result = pymetis.part_graph(nparts=2, adjacency=adj, vweights=vw)
    membership = list(result.vertex_part)

    # Pick the smaller side's boundary as the separator. (Either side
    # works; smaller side gives marginally fewer candidates.)
    counts = [membership.count(0), membership.count(1)]
    smaller_side = 0 if counts[0] <= counts[1] else 1
    other_side = 1 - smaller_side

    separator = {
        nodes[i]
        for i, side in enumerate(membership)
        if side == smaller_side
        and any(
            membership[n2i[m]] == other_side
            for m in graph.neighbors(nodes[i])
            if m in component
        )
    }
    if not separator:
        # Degenerate — fall back to one node.
        return {nodes[0]}
    return separator


def _approximate_balanced_separator(graph, component) -> Set:
    """
    Approximate balanced vertex separator of ``component`` via BFS bisection.

    Procedure:

    1. 2-BFS to find approximate diameter endpoints ``u`` and ``v``.
    2. Run a single BFS from each, in lockstep; assign each node to
       whichever endpoint reaches it first ("Voronoi" partition).
    3. The separator is the side-A nodes that have a side-B neighbour
       — a "border" between the two halves.

    Returns the border as a set. Marking these nodes as candidates
    disconnects the component into two roughly-balanced halves.

    On a planar graph of size n the border is typically O(√n); over
    log(D/τ) recursive bisections the total candidate count converges
    to the Frederickson 1987 O(n/√r) bound.
    """
    if len(component) <= 1:
        return set(component)

    s = next(iter(component))

    dist_s, _ = _bfs_with_predecessors(graph, s, component)
    u = max(dist_s, key=dist_s.get)

    dist_u, _ = _bfs_with_predecessors(graph, u, component)
    v = max(dist_u, key=dist_u.get)
    if v == u:
        return {u}

    side = {u: "u", v: "v"}
    queue_u: deque = deque([u])
    queue_v: deque = deque([v])

    while queue_u or queue_v:
        for q, label in ((queue_u, "u"), (queue_v, "v")):
            if not q:
                continue
            n = q.popleft()
            for nbr in graph.neighbors(n):
                if nbr in component and nbr not in side:
                    side[nbr] = label
                    q.append(nbr)

    border = {
        n
        for n in component
        if side.get(n) == "u"
        and any(
            side.get(nbr) == "v"
            for nbr in graph.neighbors(n)
            if nbr in component
        )
    }
    if not border:
        return {u}
    return border


def _approximate_center(graph, component):
    """
    Approximate the graph 1-center of ``component`` in O(|V| + |E|).

    Three BFS calls (Roditty-Williams 2013, also folklore):

    1. BFS from any node ``s``; let ``u`` be the farthest reached node.
    2. BFS from ``u``; let ``v`` be the farthest. The ``u-v`` path is an
       approximate diameter (exact for trees).
    3. Walk from ``v`` halfway along the BFS predecessor chain back
       toward ``u``. The midpoint approximates the 1-center.

    Exact for trees; for general graphs gives a constant-factor
    approximation that, on planar / quasi-planar graphs, is within a
    small additive constant of the true center.
    """
    s = next(iter(component))

    dist_s, _ = _bfs_with_predecessors(graph, s, component)
    u = max(dist_s, key=dist_s.get)

    dist_u, pred_u = _bfs_with_predecessors(graph, u, component)
    v = max(dist_u, key=dist_u.get)

    midpoint_steps = dist_u[v] // 2
    node = v
    for _ in range(midpoint_steps):
        node = pred_u[node]
    return node
