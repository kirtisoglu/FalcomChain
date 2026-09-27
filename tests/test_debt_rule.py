"""
Numerical checks of the debt mechanism (paper Appendix A) under the default
``rule="paper"`` window, run on the real recursion with instrumentation.

Checked on every logged extraction:
- Cor. A.2 (containment): the extracted district is globally balanced,
  d(D_r)/c in d_bar[1-eps, 1+eps];
- Thm. A.1 (debt evolution): delta^(r) lies in the stated interval;
- Cor. A.3 (safe range): with c_max <= 2 the debt never leaves [-2 tau, 2 tau];
- Cor. A.2 (feasibility): a successful extraction implies |delta^(r-1)| <= 2 tau;
- Remark A.5 (telescoping): the debt returns to 0 when the recursion closes.
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


def _recursion_log(seed, c_max, dims=(12, 12), n_cands=36, jitter=0.4):
    """Run one full recursive partitioning and return its per-extraction log."""
    set_seed(seed)
    g = Grid(dimensions=dims, num_candidates=n_cands, density="uniform").graph
    rs = random.Random(seed)
    for n in g.nodes:                       # heterogeneous integer demand in [30, 70]
        g.nodes[n]["demand"] = int(round(50 * (1 + jitter * (2 * rs.random() - 1))))
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
    assert log and all(e["rule"] == "paper" for e in log)
    return d_bar, EPS * d_bar, log


@pytest.fixture(scope="module", params=[1, 2, 3], ids=["cmax1", "cmax2", "cmax3"])
def runs(request):
    """Three feasible recursions per c_max. Seeds are tried in a fixed order,
    so the outcome is deterministic. Failures are expected to be common at
    c_max = 3 (Cor. A.3(ii): a capacity-3 extraction can leave the debt
    envelope, and only rejection restores it); they are recorded, not hidden."""
    c_max = request.param
    out, failures = [], 0
    for seed in range(60):
        r = _recursion_log(seed, c_max)
        if r is None:
            failures += 1
        else:
            out.append(r)
        if len(out) == 3:
            break
    assert len(out) == 3, f"too few feasible recursions for c_max={c_max} ({failures} failures)"
    return c_max, out


def test_rule_paper_window_scales_with_capacity():
    g = nx.path_graph(3)
    for n in g.nodes:
        g.nodes[n].update(demand=10, area=1, candidate=1)
    params = dict(ideal_demand=100.0, epsilon=0.1, capacity_level=2, n_teams=4)
    paper = SpanningTree(graph=g, params=CutParams(rule="paper", **params))
    legacy = SpanningTree(graph=g, params=CutParams(rule="per_team", **params))
    # capacity 2: paper window [180, 220]; legacy window [190, 210]
    assert paper.has_ideal_demand(2, 185) and not legacy.has_ideal_demand(2, 185)
    assert paper.has_ideal_demand(1, 105) and legacy.has_ideal_demand(1, 105)
    assert not paper.has_ideal_demand(2, 221)


def test_extracted_districts_are_globally_balanced(runs):
    _, out = runs
    for d_bar, tau, log in out:
        for e in log:
            per_team = e["demand"] / e["capacity"]
            assert d_bar * (1 - EPS) - TOL <= per_team <= d_bar * (1 + EPS) + TOL


def test_debt_evolution_matches_theorem(runs):
    _, out = runs
    for d_bar, tau, log in out:
        for e in log:
            delta, c, after = e["debt"], e["capacity"], e["debt_after"]
            if delta >= 0:
                lo, hi = delta - c * tau, (1 - c) * delta + c * tau
            else:
                lo, hi = (1 - c) * delta - c * tau, delta + c * tau
            assert lo - TOL <= after <= hi + TOL


def test_successful_extraction_implies_feasible_window(runs):
    _, out = runs
    for d_bar, tau, log in out:
        for e in log:
            assert abs(e["debt"]) <= 2 * tau + TOL


def test_safe_capacity_range_keeps_debt_in_envelope(runs):
    c_max, out = runs
    if c_max > 2:
        pytest.skip("envelope is guaranteed only for c_max <= 2")
    for d_bar, tau, log in out:
        for e in log:
            assert abs(e["debt_after"]) <= 2 * tau + TOL


def test_debt_telescopes_to_zero(runs):
    _, out = runs
    for d_bar, tau, log in out:
        assert abs(log[-1]["debt_after"]) < 1e-6 * d_bar
