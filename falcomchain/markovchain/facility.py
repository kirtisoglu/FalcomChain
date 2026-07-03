from typing import Dict, Optional, Tuple


class FacilityAssignment:
    """
    Computes and caches the best facility center and access cost for each
    district, given a ChainState.

    The optimal center is the **demand-weighted 1-median**: the candidate
    that minimises the total demand-weighted travel time

        cost(D, f) = sum_{v in D} d_v * dist(f, v)

    from the facility to every node in the district. This matches the
    base-level term of the disaggregated hierarchical median objective
    (paper Eq. for the MILP benchmark) and is the optimization analogue
    that the chain's facility-assignment rule (paper Section 5.4) targets.

    Centers and costs are computed eagerly and cached. Only districts that
    changed (via ``flow.node_flows`` and ``flow.part_flows``) are recomputed
    in :meth:`updated`.

    :ivar centers: Maps district ID → demand-weighted 1-median node.
    :ivar radii:   Maps district ID → 1-median access cost
        ``sum_v d_v * dist(center, v)`` (NOT a covering radius). The name
        is retained for backward compatibility of accessors.
    """

    __slots__ = ("_centers", "_radii", "_travel_times", "_demands")

    def __init__(self, travel_times: Dict, demands: Optional[Dict] = None) -> None:
        """
        :param travel_times: Dict keyed by ``(facility_node, node)`` → travel time.
        :param demands: Dict keyed by node → demand. Required for the
            demand-weighted 1-median; if ``None``, unit demands are assumed.
        """
        self._travel_times = travel_times
        self._demands = demands if demands is not None else {}
        self._centers: Dict = {}
        self._radii: Dict = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state) -> "FacilityAssignment":
        """
        Build a fully computed FacilityAssignment from a ChainState.
        All districts are evaluated eagerly.

        :param state: A ChainState whose assignment has ``travel_times`` set.
        """
        travel_times = state.assignment.travel_times
        graph = state.graph
        demands = {v: graph.nodes[v]["demand"] for v in graph.nodes}
        fa = cls(travel_times, demands)
        assignment = state.assignment
        for part in assignment.parts:
            fa._centers[part], fa._radii[part] = fa._compute(
                assignment.candidates[part], assignment.parts[part]
            )
        return fa

    @classmethod
    def updated(cls, previous: "FacilityAssignment", state) -> "FacilityAssignment":
        """
        Build a FacilityAssignment for ``state`` by copying the previous one
        and recomputing only the districts that changed.

        :param previous: FacilityAssignment from the parent ChainState.
        :param state:    The proposed ChainState.
        """
        fa = cls(previous._travel_times, previous._demands)
        fa._centers = previous._centers.copy()
        fa._radii = previous._radii.copy()

        flow = state.partition.flow
        assignment = state.assignment

        # Drop removed districts
        for part in flow.part_flows["out"]:
            fa._centers.pop(part, None)
            fa._radii.pop(part, None)

        # Recompute districts that gained/lost nodes or are brand new
        changed = set(flow.node_flows.keys()) | flow.part_flows["in"]
        for part in changed:
            if part in assignment.parts:
                fa._centers[part], fa._radii[part] = fa._compute(
                    assignment.candidates[part], assignment.parts[part]
                )

        return fa

    # ------------------------------------------------------------------
    # Core demand-weighted 1-median computation
    # ------------------------------------------------------------------

    def _compute(self, candidates, nodes) -> Tuple[Optional[object], float]:
        """
        Find the candidate minimising the total demand-weighted travel time
        ``sum_{v in nodes} d_v * dist(candidate, v)`` (demand-weighted
        1-median). Returns ``(best_candidate, access_cost)``.
        """
        best_candidate = None
        best_cost = float("inf")
        d = self._demands

        for candidate in candidates:
            cost = sum(
                d.get(node, 1.0) * self._travel_times[(candidate, node)]
                for node in nodes
            )
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate

        return best_candidate, best_cost

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def centers(self) -> Dict:
        return self._centers

    @property
    def radii(self) -> Dict:
        return self._radii

    def center(self, part) -> Optional[object]:
        return self._centers.get(part)

    def radius(self, part) -> float:
        return self._radii.get(part, float("inf"))

    def __repr__(self):
        return f"<FacilityAssignment [{len(self._centers)} districts]>"


def minimax_super_selector(super_id, super_candidates, base_nodes, demands,
                           travel_times):
    """
    Default level-2 facility selector: demand-weighted 1-median over the
    base-level nodes of the superdistrict.

    Picks the super-candidate minimising the total demand-weighted travel
    time to every base unit in the superdistrict — the upper-level
    coordination term of the disaggregated hierarchical median objective:

    .. math::

        \\mathrm{cost}(S, f) \\;=\\; \\sum_{v \\in V^1[S]} d_v \\, \\mathrm{d}(f, v).

    This mirrors the base-level rule (:class:`FacilityAssignment`) one level
    up: each level's facility is the demand-weighted 1-median of the units
    it serves, and the two levels are scored separably (paper Section 5.4 /
    the MILP disaggregated-median objective).

    Returns ``(best_candidate, access_cost)``, or ``(None, inf)`` if
    ``super_candidates`` or ``base_nodes`` is empty.

    :param super_id: ID of the superdistrict (informational; unused by default).
    :param super_candidates: Iterable of nodes in F^2 ∩ V(G^1[D^2]).
    :param base_nodes: Iterable of all base-level nodes in this superdistrict.
    :param demands: Dict node → demand (unit demand assumed if missing).
    :param travel_times: Dict (facility_node, node) -> travel time.

    :returns: (best_super_candidate, demand_weighted_access_cost)
    """
    best_candidate = None
    best_cost = float("inf")
    base = list(base_nodes)
    if not base:
        return None, float("inf")

    for candidate in super_candidates:
        try:
            cost = sum(
                demands.get(v, 1.0) * travel_times[(candidate, v)]
                for v in base
            )
        except KeyError:
            continue
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate

    return best_candidate, best_cost


class SuperFacilityAssignment:
    """
    Computes level-2 (super-) facility centers for each superdistrict.

    For each superdistrict D^2, the level-2 facility is selected from
    F^2 ∩ V(G^1[D^2]) — the super-candidates (``super_candidate=1`` nodes)
    that lie in the base-level subgraph induced by D^2's constituent
    level-1 districts. Selection is delegated to a pluggable callable
    (see ``selection_fn``); the default is :func:`minimax_super_selector`
    (Eq. 18).

    **Soft constraint at level 2.** If a superdistrict contains no
    super-candidates, no level-2 facility is assigned for it — the entry
    is simply absent from ``centers`` and ``radii``. The chain is not
    rejected. Callers can iterate ``state.partition.super_parts`` to
    detect superdistricts without a level-2 facility.

    :ivar centers: Maps superdistrict ID -> selected super-candidate node.
        May not contain an entry for every superdistrict (soft constraint).
    :ivar radii:   Maps superdistrict ID -> covering radius (matches the
        keys of ``centers``).
    """

    __slots__ = ("_centers", "_radii")

    def __init__(self):
        self._centers = {}
        self._radii = {}

    @classmethod
    def from_state(cls, state, selection_fn=None) -> "SuperFacilityAssignment":
        """
        Build a SuperFacilityAssignment from a ChainState.

        Iterates ``state.partition.super_parts`` (superdistrict ID -> set
        of level-1 district IDs), gathers super-candidates and base nodes
        for each, and delegates the actual selection to ``selection_fn``.

        :param state: A ChainState whose partition has ``super_assignment``
            populated and whose ``assignment.travel_times`` is set. The
            level-1 facility centers in ``state.facility`` must already
            be populated (they are the "spokes" the L2 facility serves).
        :param selection_fn: Callable
            ``(super_id, super_candidates, base_nodes, demands, travel_times) -> (node, cost)``.
            Defaults to :func:`minimax_super_selector` (demand-weighted
            1-median over base nodes).
        """
        sfa = cls()
        if selection_fn is None:
            selection_fn = minimax_super_selector

        travel_times = state.assignment.travel_times
        if travel_times is None:
            return sfa

        partition = state.partition
        super_parts = partition.super_parts  # super_id -> frozenset of level-1 IDs
        graph_nodes = partition.graph.nodes  # FrozenGraph node attr access
        demands = {v: graph_nodes[v]["demand"] for v in partition.graph.nodes}

        for super_id, l1_ids in super_parts.items():
            # Base-level nodes in V(G^1[D^2]): union over level-1 districts in D^2.
            base_nodes = set()
            for l1_id in l1_ids:
                if l1_id in partition.parts:
                    base_nodes |= partition.parts[l1_id]
            if not base_nodes:
                continue

            # Super-candidates inside this superdistrict.
            super_candidates = [
                v for v in base_nodes
                if graph_nodes[v].get("super_candidate", 0)
            ]
            if not super_candidates:
                # Soft constraint: no level-2 facility for this superdistrict.
                continue

            best, cost = selection_fn(
                super_id, super_candidates, base_nodes, demands, travel_times
            )
            if best is not None:
                sfa._centers[super_id] = best
                sfa._radii[super_id] = cost

        return sfa

    @classmethod
    def updated(
        cls,
        previous: "SuperFacilityAssignment",
        state,
        selection_fn=None,
    ) -> "SuperFacilityAssignment":
        """
        Build a SuperFacilityAssignment incrementally from ``previous``.

        Mirrors :meth:`FacilityAssignment.updated` at level 2: copies
        ``previous``'s ``centers`` and ``radii`` and recomputes only the
        super-districts that changed this step. Unchanged super-districts
        keep their cached center / radius.

        Cheaper than :meth:`from_state` when the partition's ``superflip``
        only touches a few super-districts (which is the common case under
        ``hierarchical_recom`` — exactly one super-district is merged and
        re-partitioned per step).

        :param previous: The parent state's ``SuperFacilityAssignment``.
        :param state: The proposed :class:`ChainState`.
        :param selection_fn: See :meth:`from_state`.
        """
        sfa = cls()
        if selection_fn is None:
            selection_fn = minimax_super_selector

        partition = state.partition
        travel_times = state.assignment.travel_times
        if travel_times is None:
            return sfa

        # Carry forward previous entries so the incremental path keeps
        # the cached values for unchanged super-districts.
        sfa._centers = previous._centers.copy()
        sfa._radii = previous._radii.copy()

        # Diff: which super-districts changed?
        # `superflip.merged_ids` are the *parent's* super-IDs that were
        # consumed; `superflip.flips` is the new super_assignment, so the
        # *new* super-IDs are the unique values among the new assignment.
        superflip = partition.superflip
        new_super_assignment = (
            superflip.flips if superflip is not None else {}
        )
        new_super_ids = set(new_super_assignment.values())
        merged_ids = (
            set(superflip.merged_ids) if superflip is not None else set()
        )

        # Drop super-IDs that no longer exist (parent's super-IDs that
        # were merged this step but are NOT among the new super-IDs).
        for sid in merged_ids:
            if sid not in new_super_ids:
                sfa._centers.pop(sid, None)
                sfa._radii.pop(sid, None)

        # Recompute super-IDs that are *new* this step (typically the IDs
        # produced by the level-2 re-partition for the merged region).
        # An "unchanged" super-district has the same ID and its constituent
        # L1 districts haven't changed; its center is still valid.
        super_parts = partition.super_parts
        graph_nodes = partition.graph.nodes
        demands = {v: graph_nodes[v]["demand"] for v in partition.graph.nodes}
        recompute_ids = (new_super_ids - sfa._centers.keys()) | merged_ids
        # Also recompute any super-district whose L1 footprint changed
        # (a parent super-ID that was kept but had its L1 districts
        # remapped — rare but possible).
        for sid in recompute_ids & new_super_ids:
            l1_ids = super_parts.get(sid, frozenset())
            base_nodes = set()
            for l1_id in l1_ids:
                if l1_id in partition.parts:
                    base_nodes |= partition.parts[l1_id]
            if not base_nodes:
                sfa._centers.pop(sid, None)
                sfa._radii.pop(sid, None)
                continue
            super_candidates = [
                v for v in base_nodes
                if graph_nodes[v].get("super_candidate", 0)
            ]
            if not super_candidates:
                sfa._centers.pop(sid, None)
                sfa._radii.pop(sid, None)
                continue
            best, cost = selection_fn(
                sid, super_candidates, base_nodes, demands, travel_times
            )
            if best is not None:
                sfa._centers[sid] = best
                sfa._radii[sid] = cost
            else:
                sfa._centers.pop(sid, None)
                sfa._radii.pop(sid, None)

        return sfa

    @property
    def centers(self):
        return self._centers

    @property
    def radii(self):
        return self._radii

    def center(self, part):
        return self._centers.get(part)

    def radius(self, part):
        return self._radii.get(part, float("inf"))

    def __repr__(self):
        return f"<SuperFacilityAssignment [{len(self._centers)} superdistricts]>"
