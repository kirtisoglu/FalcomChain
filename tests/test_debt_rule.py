"""
Numerical checks of the debt mechanism on the instrumented recursion under
the default balance rule (``rule="per_team"``: absolute per-team half-width,
debt-clipped window).

With the per-team window [L~, U~] clipped by the debt delta (delta >= 0:
L~ = d_bar - tau, U~ = d_bar + tau - delta), a district at capacity c is
admissible iff  |d(D) - c d'| <= (U~ - L~)/2  with d' = (L~ + U~)/2. Hence

    delta' = delta + d(D) - c d_bar  in  [ (3-c)/2 delta - tau, (1-c)/2 delta + tau ]

for delta >= 0 (mirror image for delta < 0), which gives the invariant
|delta| <= tau for every c <= 3 -- the reason a recursion with c_max <= 3
(the London calibration) never meets an empty window. The checks below verify,
on every logged extraction:
- containment: the extracted district is globally balanced,
  d(D)/c in d_bar[1-eps, 1+eps];
- the debt-evolution interval above;
- the invariant |delta| <= tau (c_max in {1, 2, 3});
- telescoping: the debt returns to 0 when the recursion closes.
The ``"scaled"`` rule (capacity-scaled window) is covered by a unit test of
the window itself and by an informational feasibility comparison.
"""
import random

import networkx as nx
import pytest

import falcomchain.tree.tree as T
from falcomchain.graph.grid import Grid
from falcomchain.random import set_seed
from falcomchain.tree.errors import CutSearchExhausted
from falcomchain.tree.tree import CutParams, SpanningTree, capacitated_recursive_tree

EPS = 0.15
TOL = 1e-6


def _grid(seed, dims=(12, 12), n_cands=36, jitter=0.4):
    set_seed(seed)
    g = Grid(dimensions=dims, num_candidates=n_cands, density="uniform").graph
    rs = random.Random(seed)
    for n in g.nodes:                       # heterogeneous integer demand in [30, 70]
        g.nodes[n]["demand"] = int(round(50 * (1 + jitter * (2 * rs.random() - 1))))
    return g


def _recursion_log(seed, c_max):
    """Run one full recursive partitioning and return its per-extraction log."""
    g = _grid(seed)
    total = sum(d["demand"] for _, d in g.nodes(data=True))
    n_teams = 12
    d_bar = total / n_teams                 # local target -> debt telescopes to 0
    T.DEBT_LOG_ENABLED = True
    T.DEBT_LOG.clear()
    try:
        capacitated_recursive_tree(
            graph=g, n_teams=n_teams, demand_target=d_bar, epsilon=EPS,
            capacity_level=c_max, max_attempts=1000,
        )
    except CutSearchExhausted:
        return None
    finally:
        T.DEBT_LOG_ENABLED = False
    log = [dict(e) for e in T.DEBT_LOG if "capacity" in e and not e["supergraph"]]
    assert log and all(e["rule"] == "per_team" for e in log)
    return d_bar, EPS * d_bar, log


@pytest.fixture(scope="module", params=[1, 2, 3], ids=["cmax1", "cmax2", "cmax3"])
def runs(request):
    """Four feasible recursions per c_max; seeds are tried in a fixed order so
    the outcome is deterministic. The number of infeasible seeds is recorded."""
    c_max = request.param
    out, failures = [], 0
    for seed in range(40):
        r = _recursion_log(seed, c_max)
        if r is None:
            failures += 1
        else:
            out.append(r)
        if len(out) == 4:
            break
    assert len(out) == 4, f"too few feasible recursions for c_max={c_max} ({failures} failures)"
    return c_max, out


def test_scaled_rule_window_scales_with_capacity():
    g = nx.path_graph(3)
    for n in g.nodes:
        g.nodes[n].update(demand=10, area=1, candidate=1)
    params = dict(ideal_demand=100.0, epsilon=0.1, capacity_level=2, n_teams=4)
    scaled = SpanningTree(graph=g, params=CutParams(rule="scaled", **params))
    absolute = SpanningTree(graph=g, params=CutParams(rule="per_team", **params))
    # capacity 2: scaled window [180, 220]; absolute window [190, 210]
    assert scaled.has_ideal_demand(2, 185) and not absolute.has_ideal_demand(2, 185)
    assert scaled.has_ideal_demand(1, 105) and absolute.has_ideal_demand(1, 105)
    assert not scaled.has_ideal_demand(2, 221) and not absolute.has_ideal_demand(2, 211)


def test_extracted_districts_are_globally_balanced(runs):
    _, out = runs
    for d_bar, tau, log in out:
        for e in log:
            per_team = e["demand"] / e["capacity"]
            assert d_bar * (1 - EPS) - TOL <= per_team <= d_bar * (1 + EPS) + TOL


def test_debt_evolution_interval(runs):
    _, out = runs
    for d_bar, tau, log in out:
        for e in log:
            delta, c, after = e["debt"], e["capacity"], e["debt_after"]
            if delta >= 0:
                lo, hi = (3 - c) / 2 * delta - tau, (1 - c) / 2 * delta + tau
            else:
                lo, hi = (1 - c) / 2 * delta - tau, (3 - c) / 2 * delta + tau
            assert lo - TOL <= after <= hi + TOL


def test_debt_stays_within_one_tolerance(runs):
    c_max, out = runs
    assert c_max <= 3
    for d_bar, tau, log in out:
        for e in log:
            assert abs(e["debt"]) <= tau + TOL
            assert abs(e["debt_after"]) <= tau + TOL


def test_debt_telescopes_to_zero(runs):
    _, out = runs
    for d_bar, tau, log in out:
        assert abs(log[-1]["debt_after"]) < 1e-6 * d_bar


def test_absolute_rule_closes_at_least_as_often_as_scaled():
    """Informational comparison at c_max = 3 over the first 12 seeds; the
    absolute rule should rarely fail while the scaled rule often does."""
    counts = {}
    for rule in ("per_team", "scaled"):
        ok = 0
        for seed in range(12):
            g = _grid(seed)
            total = sum(d["demand"] for _, d in g.nodes(data=True))
            try:
                capacitated_recursive_tree(graph=g, n_teams=12, demand_target=total / 12,
                                           epsilon=EPS, capacity_level=3,
                                           max_attempts=300, rule=rule)
                ok += 1
            except CutSearchExhausted:
                pass
        counts[rule] = ok
    print(f"\nfeasible recursions out of 12 at c_max=3: {counts}")
    assert counts["per_team"] >= counts["scaled"]
