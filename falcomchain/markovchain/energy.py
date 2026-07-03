from falcomchain.markovchain.state import ChainState


def compute_energy(state: ChainState) -> float:
    """
    Disaggregated hierarchical median energy (the FalCom optimization
    objective; matches the MILP benchmark objective of the experiments
    section):

        E(s) = Σ_D Σ_{v∈D} d_v·dist(v, f¹(D))        (base-level access)
             + Σ_S Σ_{v∈V¹[S]} d_v·dist(v, f²(S))     (upper-level coordination)

    where D ranges over level-1 districts with level-1 facility f¹(D),
    S ranges over superdistricts with level-2 facility f²(S), d_v is node
    demand, and dist is read from ``Assignment.travel_times``. The two
    terms are separable across levels: each facility is the
    demand-weighted 1-median of the units it serves.

    The level-2 term is included only when ``state.super_facility`` is
    populated (i.e. a ``super_facility_fn`` was supplied); otherwise the
    energy is the base-level access term alone.

    :param state: The current chain state.
    :returns: Disaggregated median (lower is better).
    """
    assignment = state.assignment
    travel_times = assignment.travel_times
    graph = state.graph
    centers = state.facility.centers

    # ---- base-level access term ----
    total = 0.0
    for district, nodes in assignment.parts.items():
        center = centers.get(district)
        if center is None:
            continue
        for node in nodes:
            demand = graph.nodes[node]["demand"]
            total += demand * travel_times[(center, node)]

    # ---- upper-level coordination term ----
    sf = state.super_facility
    if sf is not None:
        super_centers = sf.centers
        super_parts = state.partition.super_parts  # super_id -> L1 district IDs
        for super_id, l1_ids in super_parts.items():
            center2 = super_centers.get(super_id)
            if center2 is None:
                continue
            for did in l1_ids:
                for node in assignment.parts.get(did, ()):
                    demand = graph.nodes[node]["demand"]
                    total += demand * travel_times[(center2, node)]

    return total


def compute_energy_fairness(state: ChainState) -> float:
    """
    Two-level fairness energy: sum of squared deviations from the mean,
    at level 1 (district covering radii) and level 2 (L1-to-super
    facility travel times).

    .. math::

        E(s) \\;=\\; \\sum_{i \\in P^{1}} \\bigl(r_{i}^{1} - \\bar{r}^{1}\\bigr)^{2}
            + \\sum_{i \\in P^{1}} \\bigl(t(f_{i}^{1}, f_{P^{2}(i)}^{2}) - \\bar{t}\\bigr)^{2}

    where:
      - :math:`r_{i}^{1}` is the L1 covering radius of district *i*,
        i.e. the max travel time from any unit in *i* to its facility,
        as cached in :attr:`state.facility.radii`.
      - :math:`t(f_{i}^{1}, f_{P^{2}(i)}^{2})` is the travel time from
        district *i*'s L1 facility to the super-facility of its parent
        super-district.
      - :math:`\\bar{r}^{1}` and :math:`\\bar{t}` are the means across
        all districts in the partition.

    Lower :math:`E(s)` ⇒ more equitable coverage at both levels: every
    district is similarly well covered, and every district's L1 hub is
    similarly close to its sector HQ. This is an *equity* objective,
    distinct from the standard EMS minisum used by
    :func:`compute_energy`.

    :param state: The current chain state.
    :type state: ChainState
    :returns: Sum of squared deviations from the mean across both
        levels (units: minutes squared if travel times are in minutes).
    :rtype: float
    """
    radii_l1 = state.facility.radii if state.facility else {}
    centers_l1 = state.facility.centers if state.facility else {}
    centers_l2 = (
        state.super_facility.centers if state.super_facility else {}
    )
    super_assignment = state.partition.super_assignment
    travel = state.assignment.travel_times

    # L1 term: variance of district covering radii
    r_values = [r for r in radii_l1.values() if r != float("inf")]
    if not r_values:
        return 0.0
    mean_r = sum(r_values) / len(r_values)
    term1 = sum((r - mean_r) ** 2 for r in r_values)

    # L2 term: variance of L1-to-L2 facility travel times.
    # One sample per district: the edge from its L1 centre to the L2
    # centre of its super-district.
    t_values = []
    for district_id, super_id in super_assignment.items():
        f1 = centers_l1.get(district_id)
        f2 = centers_l2.get(super_id)
        if f1 is None or f2 is None:
            continue
        t = travel[(f1, f2)]
        if t == float("inf"):
            continue
        t_values.append(t)
    if t_values:
        mean_t = sum(t_values) / len(t_values)
        term2 = sum((t - mean_t) ** 2 for t in t_values)
    else:
        term2 = 0.0

    return term1 + term2


def compute_energy_eccentricity(state: ChainState) -> float:
    """
    Hierarchical sum of squared eccentricities — the energy
    counterpart of ψ that encodes both efficiency and equity by
    construction.

    .. math::

        E_{\\mathrm{ecc}}(s) \\;=\\; \\sum_{i \\in P^{1}} \\bigl(r_{i}^{1}\\bigr)^{2}
            + \\sum_{S \\in P^{2}} \\bigl(r_{S}^{2}\\bigr)^{2}

    where:

    - :math:`r_{i}^{1} = \\max_{v \\in D_{i}} t(f_{i}^{1}, v)` is the
      eccentricity of district *i*'s **L1 facility** over the
      **base-graph units** in :math:`D_i` — exactly the
      :math:`g(T_{u})` that ψ uses at cut time. Cached in
      :attr:`state.facility.radii`.

    - :math:`r_{S}^{2} = \\max_{i \\in S} t(f_{S}^{2}, f_{i}^{1})` is the
      eccentricity of super-district *S*'s **super-facility** over the
      **L1 facilities** it contains. The hierarchy is what motivates
      this: a super-facility coordinates L1 hubs, not base units
      directly, so the L2 service relation is :math:`f^{2} \\to f^{1}`.

    The squared sum decomposes by the identity

    .. math::

        \\sum_i r_i^{2} \\;=\\; n \\bar r^{2} \\;+\\; \\sum_i (r_i - \\bar r)^{2},

    so minimising it pushes the *mean* covering radius down
    (efficiency) AND tightens the *spread* across districts (access
    equity) in matched units. Locally decomposable: each L1 cut
    contributes one :math:`(r_i^{1})^{2}` term, so the per-cut ψ score
    :math:`\\psi(T_{u}) = \\phi \\cdot \\exp(-\\gamma \\, g(T_{u}))`
    pushes directly against the quantity that appears in the energy.

    :param state: The current chain state.
    :type state: ChainState
    :returns: Sum of squared L1 and L2 eccentricities (units: minutes²
        if travel times are in minutes).
    :rtype: float
    """
    r1 = state.facility.radii if state.facility else {}
    term1 = sum(r * r for r in r1.values() if r != float("inf"))

    # L2 term: eccentricity of the chain-chosen super-facility over
    # the L1 facilities in its super-district. Uses
    # ``state.super_facility.centers`` so the energy reflects the
    # chain's actual super-facility-assignment rule (which may be
    # global, in-subtree, or fixed depending on the application).
    term2 = 0.0
    sf = state.super_facility
    if sf is not None:
        super_centers = sf.centers
        l1_centers = state.facility.centers if state.facility else {}
        super_parts = state.partition.super_parts
        travel_times = state.assignment.travel_times
        if travel_times is not None:
            for super_id, district_ids in super_parts.items():
                f2 = super_centers.get(super_id)
                if f2 is None:
                    continue
                ecc = 0.0
                missing = False
                for pid in district_ids:
                    f1 = l1_centers.get(pid)
                    if f1 is None:
                        missing = True
                        break
                    t = travel_times[(f2, f1)]
                    if t == float("inf"):
                        missing = True
                        break
                    if t > ecc:
                        ecc = t
                if not missing:
                    term2 += ecc * ecc
    return term1 + term2


def compute_plan_summary(
    state: ChainState,
    *,
    coverage_thresholds=(8.0, 15.0),
    p90: float = 0.90,
    demand_target: float | None = None,
    demand_tolerance: float | None = None,
) -> dict:
    """
    Per-plan operational + FalCom-radii summary for any ``ChainState``.

    Read-only aggregator over the partition + facility assignments +
    ``Assignment.travel_times``. Works for:

      * a state produced by the chain (``hierarchical_recom``),
      * an initial state from :meth:`Partition.from_random_assignment`,
      * a benchmark state built manually (e.g. a Voronoi imitation of
        an operational layout).

    All travel times are read from ``state.assignment.travel_times`` —
    units are whatever the travel-time table is in (typically minutes
    when populated from proxies or FalcomTravel).

    Returns
    -------
    dict with three keys:

      * ``per_district`` — :pyobj:`dict[district_id, dict]` with
        ``n_nodes``, ``demand``, ``l1_radius`` (max travel time from
        any unit in the district to its L1 facility, == FalCom's
        :math:`g(T_u)` once the cut becomes the district),
        ``mean_t`` (demand-weighted mean travel time within the
        district), ``t_p90`` (demand-weighted p-th percentile),
        ``coverage_<th>`` (fraction of demand within ``<th>`` units of
        time for each threshold), ``center`` and ``teams``.

      * ``per_super`` — :pyobj:`dict[super_id, dict]` with
        ``n_districts``, ``demand``, ``l2_radius`` (max travel time
        from the super-facility to any **L1 facility** in the
        super-district, i.e. paper Eq. 27 with the chain's chosen
        super-facility), ``mean_l1_radius``, ``super_center`` and
        ``teams``.

      * ``global`` — aggregates: ``n_districts``, ``n_super``,
        ``total_demand``, ``T_mean`` (demand-weighted across all
        nodes), ``T_p90``, ``coverage_<th>``, ``E_minisum``
        (== :func:`compute_energy`), ``E_eccentricity`` (==
        :func:`compute_energy_eccentricity`), ``max_l1_radius``,
        ``mean_l1_radius``, ``max_l2_radius``.

    Parameters
    ----------
    state : :class:`ChainState`
    coverage_thresholds : tuple[float, ...]
        Time thresholds (in the travel-time units) for the coverage
        statistics — default ``(8, 15)`` matches NHS England Cat-1
        targets.
    p90 : float
        Percentile target for the demand-weighted percentile column
        (default 0.90).

    Notes
    -----
    ``l2_radius`` uses the chain's current super-facility assignment
    (``state.super_facility.centers``). If the chain's super-facility
    rule is global-minimax-over-base-nodes (the LAS convention) the
    radius here may exceed the paper-Eq.-27 minimum-over-candidates;
    that's intentional — the function reports what the chain *did*,
    not what an ideal super-facility selector would do.
    """
    partition = state.partition
    parts = partition.parts
    teams = partition.teams
    travel = state.assignment.travel_times
    graph_nodes = (partition.graph.nodes
                   if hasattr(partition.graph, "nodes")
                   else partition.graph.graph.nodes)

    centers_l1 = state.facility.centers if state.facility else {}
    super_parts = partition.super_parts
    centers_l2 = state.super_facility.centers if state.super_facility else {}

    def _percentile(rows, target_frac):
        if not rows:
            return None
        total = sum(d for _, d in rows)
        target = target_frac * total
        cum = 0
        for t, d in rows:
            cum += d
            if cum >= target:
                return float(t)
        return float(rows[-1][0])

    def _coverage(rows, threshold):
        if not rows:
            return None
        total = sum(d for _, d in rows)
        if not total:
            return 0.0
        return sum(d for t, d in rows if t <= threshold) / total

    # ---- per-district --------------------------------------------------
    per_district = {}
    all_rows = []
    for pid, nodes in parts.items():
        f = centers_l1.get(pid)
        if f is None:
            continue
        rows = []
        demand_tot = 0
        for v in nodes:
            d = int(graph_nodes[v]["demand"])
            try:
                t = float(travel[(f, v)])
            except (KeyError, TypeError):
                continue
            if t == float("inf"):
                continue
            rows.append((t, d))
            demand_tot += d
        rows.sort()
        if not rows:
            continue
        l1_radius = max(t for t, _ in rows)
        mean_t = (sum(t * d for t, d in rows) / demand_tot) if demand_tot else 0.0
        per_district[pid] = {
            "n_nodes": len(nodes),
            "demand": demand_tot,
            "l1_radius": round(l1_radius, 4),
            "mean_t": round(mean_t, 4),
            "t_p90": round(_percentile(rows, p90) or 0.0, 4),
            "center": f,
            "teams": int(teams.get(pid, 0)),
        }
        for th in coverage_thresholds:
            per_district[pid][f"coverage_{th:g}"] = round(
                _coverage(rows, th) or 0.0, 4,
            )
        all_rows.extend(rows)

    # ---- capacity / demand balance (optional) --------------------------
    # When demand_target (w) and demand_tolerance (eps) are provided,
    # decorate each district with the FalCom feasibility band
    # [(1-eps)*teams*w, (1+eps)*teams*w] and a status flag, and roll up
    # global feasibility counts. Skipped if either param is None — the
    # balance fields then do not appear in the output dict.
    if demand_target is not None and demand_tolerance is not None:
        n_ok = n_below = n_above = n_no_teams = 0
        demand_balanced = demand_imbalanced = 0
        ratios = []
        for pid, info in per_district.items():
            teams_i = info["teams"]
            demand_i = info["demand"]
            if teams_i <= 0:
                info["balance_status"] = "no_teams"
                info["balance_ratio"] = None
                info["target_demand"] = 0.0
                info["lower_band"] = 0.0
                info["upper_band"] = 0.0
                n_no_teams += 1
                demand_imbalanced += demand_i
                continue
            target = float(teams_i) * float(demand_target)
            lower = (1.0 - demand_tolerance) * target
            upper = (1.0 + demand_tolerance) * target
            ratio = demand_i / target if target > 0 else None
            if demand_i < lower:
                status = "below"
                n_below += 1
                demand_imbalanced += demand_i
            elif demand_i > upper:
                status = "above"
                n_above += 1
                demand_imbalanced += demand_i
            else:
                status = "ok"
                n_ok += 1
                demand_balanced += demand_i
            info["target_demand"] = round(target, 2)
            info["lower_band"] = round(lower, 2)
            info["upper_band"] = round(upper, 2)
            info["balance_ratio"] = round(ratio, 4) if ratio is not None else None
            info["balance_status"] = status
            if ratio is not None:
                ratios.append(ratio)
        balance_global = {
            "demand_target": float(demand_target),
            "demand_tolerance": float(demand_tolerance),
            "n_balanced": n_ok,
            "n_below": n_below,
            "n_above": n_above,
            "n_no_teams": n_no_teams,
            "demand_balanced": demand_balanced,
            "demand_imbalanced": demand_imbalanced,
        }
        if ratios:
            balance_global["max_overload_ratio"] = round(max(ratios), 4)
            balance_global["min_underload_ratio"] = round(min(ratios), 4)
    else:
        balance_global = None

    # ---- per-super -----------------------------------------------------
    per_super = {}
    for sid, dist_ids in super_parts.items():
        f2 = centers_l2.get(sid)
        l1_rs = []
        demand_tot = 0
        teams_tot = 0
        l1_centers_here = []
        for pid in dist_ids:
            if pid in per_district:
                l1_rs.append(per_district[pid]["l1_radius"])
                demand_tot += per_district[pid]["demand"]
                teams_tot += per_district[pid]["teams"]
            f1 = centers_l1.get(pid)
            if f1 is not None:
                l1_centers_here.append(f1)
        l2_radius = None
        if f2 is not None and l1_centers_here:
            try:
                rs = [float(travel[(f2, f1)]) for f1 in l1_centers_here
                      if travel[(f2, f1)] != float("inf")]
                if rs:
                    l2_radius = max(rs)
            except (KeyError, TypeError):
                l2_radius = None
        per_super[sid] = {
            "n_districts": len(dist_ids),
            "demand": demand_tot,
            "teams": teams_tot,
            "mean_l1_radius": round(sum(l1_rs) / len(l1_rs), 4) if l1_rs else None,
            "l2_radius": round(l2_radius, 4) if l2_radius is not None else None,
            "super_center": f2,
        }

    # ---- global --------------------------------------------------------
    all_rows.sort()
    total_demand = sum(d for _, d in all_rows)
    T_mean = (sum(t * d for t, d in all_rows) / total_demand) if total_demand else 0.0
    glob = {
        "n_districts": len(per_district),
        "n_super": len(per_super),
        "total_demand": total_demand,
        "T_mean": round(T_mean, 4),
        "T_p90": round(_percentile(all_rows, p90) or 0.0, 4),
        "E_minisum": float(compute_energy(state)),
        "E_eccentricity": float(compute_energy_eccentricity(state)),
    }
    for th in coverage_thresholds:
        glob[f"coverage_{th:g}"] = round(_coverage(all_rows, th) or 0.0, 4)
    l1_all = [v["l1_radius"] for v in per_district.values()]
    if l1_all:
        glob["max_l1_radius"] = round(max(l1_all), 4)
        glob["mean_l1_radius"] = round(sum(l1_all) / len(l1_all), 4)
    l2_all = [v["l2_radius"] for v in per_super.values() if v["l2_radius"] is not None]
    glob["max_l2_radius"] = round(max(l2_all), 4) if l2_all else None
    if balance_global is not None:
        glob["balance"] = balance_global
    return {
        "per_district": per_district,
        "per_super": per_super,
        "global": glob,
    }


def compute_artificial_facility_penalty(
    state: ChainState,
    lambda_pen: float = 1.0,
    artificial_attr: str = "candidate_artificial",
) -> float:
    """
    Penalty for districts whose chosen facility is an *artificial*
    candidate (added by ``repair_facility_density``).

    Returns ``lambda_pen × (number of districts whose facility center
    has artificial_attr=1)``.

    This is the FalCom-appropriate soft penalty for the LAS-style
    case where the candidate set has been padded by repair: every
    state is structurally feasible (every district has ≥1 candidate,
    enforced by ``bipartition_tree``), but some districts use
    artificial candidates while others use real ones. With this term
    added to the energy and ``boltzmann`` acceptance, the chain
    prefers partitions where real stations carry more districts.

    Combine into a custom energy:

    .. code-block:: python

        from functools import partial
        from falcomchain.markovchain.energy import (
            compute_energy, compute_artificial_facility_penalty,
        )

        def energy_with_artificial_penalty(state):
            return (
                compute_energy(state)
                + compute_artificial_facility_penalty(state, lambda_pen=10000.0)
            )

        chain = MarkovChain(
            ..., accept=boltzmann,
            initial_state=ChainState.initial(
                ..., energy_fn=energy_with_artificial_penalty,
            ),
        )

    Choose ``lambda_pen`` so the penalty per artificial-facility
    district is comparable to a meaningful fraction of the
    demand-weighted travel time term — typically the mean district
    energy. With ``lambda_pen = 0`` the chain ignores artificial vs
    real (uniform). With ``lambda_pen = ∞`` the chain only visits
    states using all-real facilities (if such states exist).

    .. note::

        This penalty acts on *which candidate is used* per district,
        not on whether the proposal is feasible. For graph-level
        feasibility violations (Assumption 6.1), the candidate set
        must be repaired *before* the chain starts —
        ``soft_constraint_accept`` cannot help because the proposal
        mechanism never produces facility-free districts.

    :param state: The current chain state.
    :type state: ChainState
    :param lambda_pen: Penalty per district using an artificial
        candidate.
    :type lambda_pen: float
    :param artificial_attr: Node attribute marking artificial
        candidates (default ``"candidate_artificial"``, set by
        ``repair_facility_density``).
    :type artificial_attr: str
    :returns: Total artificial-facility penalty for the state.
    :rtype: float
    """
    if lambda_pen == 0.0:
        return 0.0

    graph = state.graph
    centers = state.facility.centers
    n_artificial = sum(
        1
        for center in centers.values()
        if center is not None
        and graph.nodes[center].get(artificial_attr, 0)
    )
    return lambda_pen * n_artificial


def compute_energy_delta(proposed: ChainState, current: ChainState) -> float:
    """
    Compute E(proposed) - E(current) efficiently by re-evaluating only
    the districts that changed (those in ``proposed.partition.flow.node_flows``).

    :param proposed: The proposed chain state.
    :param current:  The current chain state.
    :returns: Delta energy E(s') - E(s).
    :rtype: float
    """
    assignment = proposed.assignment
    travel_times = assignment.travel_times
    graph = proposed.graph

    flow = proposed.partition.flow
    changed_parts = set(flow.node_flows.keys()) if flow.node_flows else set()
    # Also include parts that gained/lost district IDs
    changed_parts |= flow.part_flows["in"] | flow.part_flows["out"]

    def district_energy(parts_dict, centers_dict, district):
        center = centers_dict.get(district)
        if center is None:
            return 0.0
        return sum(
            graph.nodes[node]["demand"] * travel_times[(center, node)]
            for node in parts_dict.get(district, frozenset())
        )

    delta = 0.0
    proposed_centers = proposed.facility.centers
    current_centers = current.facility.centers
    current_parts = current.assignment.parts

    for district in changed_parts:
        new_e = district_energy(assignment.parts, proposed_centers, district)
        old_e = district_energy(current_parts, current_centers, district)
        delta += new_e - old_e

    return delta
