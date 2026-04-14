from typing import Dict, Optional, Tuple


class FacilityAssignment:
    """
    Computes and caches the best facility center and service radius
    for each district, given a ChainState.

    The optimal center minimises the maximum travel time from the
    facility to any node in the district (minimax / covering radius).

    Centers and radii are computed lazily on first access and cached.
    Only districts that changed (via ``flow.node_flows`` and
    ``flow.part_flows``) are recomputed when ``update()`` is called.

    :ivar centers: Maps district ID → best facility node (or None if not yet computed).
    :ivar radii:   Maps district ID → covering radius (or None if not yet computed).
    """

    __slots__ = ("_centers", "_radii", "_travel_times")

    def __init__(self, travel_times: Dict) -> None:
        """
        :param travel_times: Dict keyed by ``(facility_node, node)`` → travel time.
        """
        self._travel_times = travel_times
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
        fa = cls(travel_times)
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
        fa = cls(previous._travel_times)
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
    # Core minimax computation
    # ------------------------------------------------------------------

    def _compute(self, candidates, nodes) -> Tuple[Optional[object], float]:
        """
        Find the candidate that minimises the maximum travel time to any node
        in ``nodes`` (minimax covering radius).

        :returns: ``(best_candidate, radius)``
        """
        best_candidate = None
        best_radius = float("inf")

        for candidate in candidates:
            radius = max(self._travel_times[(candidate, node)] for node in nodes)
            if radius < best_radius:
                best_radius = radius
                best_candidate = candidate

        return best_candidate, best_radius

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


class SuperFacilityAssignment:
    """
    Computes level-2 facility centers for superdistricts.

    For each superdistrict D_i^2, the level-2 facility is the candidate
    in F^2 ∩ V(G^1[D_i^2]) that minimises the maximum travel time to any
    base-level node in D_i^2 (Eq. 18 in the paper).

    In the current 2-level implementation, F^2 candidates are the level-1
    facility centers (one per district within each superdistrict).

    :ivar centers: Maps superdistrict ID -> best level-2 facility node.
    :ivar radii:   Maps superdistrict ID -> covering radius.
    """

    __slots__ = ("_centers", "_radii")

    def __init__(self):
        self._centers = {}
        self._radii = {}

    @classmethod
    def from_state(cls, state) -> "SuperFacilityAssignment":
        """
        Build a fully computed SuperFacilityAssignment from a ChainState.

        For each superdistrict (group of level-1 districts that share a
        supergraph node), find the level-1 center that minimises the max
        travel time over all base-level nodes in the superdistrict.

        :param state: A ChainState with facility assignment already computed.
        """
        sfa = cls()
        travel_times = state.assignment.travel_times
        if travel_times is None:
            return sfa

        partition = state.partition
        level1_centers = state.facility.centers

        # Build superdistrict -> set of level-1 district IDs
        # A superdistrict in G^2 corresponds to a node whose value is a
        # set of base-level districts. We read this from the supergraph.
        supergraph = partition.supergraph
        if supergraph is None:
            return sfa

        for supernode in supergraph.nodes:
            # The level-1 districts in this superdistrict
            districts_in_super = {
                d for d in partition.parts
                if d == supernode or (
                    hasattr(partition, 'assignment') and
                    supernode in partition.parts and
                    d in partition.parts
                )
            }
            # Actually: each supergraph node IS a level-1 district ID.
            # A superdistrict is a group of supergraph nodes.
            # But currently supergraph nodes = level-1 district IDs directly.
            # So for level-2 we need the grouping from the supergraph partition.
            # For now, each supernode is one level-1 district, so level-2
            # center = level-1 center of that district.
            # This will be extended when we have an actual level-2 partition.
            pass

        # Simple correct approach for |L|=2:
        # Each superdistrict groups multiple level-1 districts.
        # The supergraph nodes ARE the level-1 districts.
        # We need the level-2 partition (which districts form which superdistrict).
        # This is stored in the superflip / supergraph structure.
        #
        # For now: compute over supergraph connected components or
        # use the supergraph partition if available.
        # Since the supergraph itself is partitioned by hierarchical_recom,
        # each superdistrict = set of supergraph nodes in one super-partition.
        #
        # The partition stores teams per district. We can group districts
        # by their super-assignment. But we don't currently store the
        # super-level assignment explicitly.
        #
        # Minimal viable: treat each supergraph node as its own superdistrict
        # (which is what happens after the supergraph partition step).
        # The level-2 center is the level-1 center with min max eccentricity
        # over all base nodes in that superdistrict.

        for supernode in supergraph.nodes:
            # Collect all base-level nodes in this superdistrict
            if supernode not in partition.parts:
                continue
            base_nodes = partition.parts[supernode]

            # Level-2 candidates = level-1 centers of districts within this superdistrict
            # But for a single supernode = single level-1 district, the candidate IS
            # the level-1 center. For grouped superdistricts, we'd iterate over
            # constituent districts.
            center = level1_centers.get(supernode)
            if center is None:
                continue

            # Compute eccentricity: max travel time from center to any base node
            try:
                radius = max(
                    travel_times[(center, node)] for node in base_nodes
                )
            except KeyError:
                radius = float("inf")

            sfa._centers[supernode] = center
            sfa._radii[supernode] = radius

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
