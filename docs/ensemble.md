---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Ensemble Analysis

```{admonition} Goal of this page
:class: tip
Turn a chain's samples into decisions: accumulate boundary, facility,
and capacity statistics with `EnsembleStats`, check convergence, and
read the resulting maps.
```

FalCom is the first MCMC sampler for facility location. The ensemble
module turns that sampler into a decision-support tool by analyzing the
*distribution* of near-optimal designs, not just a single optimum.

## What it measures

For each sample drawn from the chain, the module accumulates:

- **Boundary frequency** — how often each edge is a district boundary.
  High = robust boundary, low = fragile.
- **Facility stability** — how often each candidate is selected as a
  facility center. High = essential, low = substitutable.
- **Capacity utilization** — per-sample coefficient of variation of
  demand-per-team and covering-radius statistics, accumulated with
  Welford's online algorithm.

## A small grid example

We load the shared 10×10 demo grid (the same dataset used by the
[Candidate Feasibility](feasibility.md) and [Level-2 Facilities](super_facility.md)
pages). The grid has **heterogeneous demand**: 5 "city center" nodes
with high demand (700), 95 "rural" nodes with low demand (16).
Heterogeneity is what makes ensemble analysis informative — without
it, every partition is exchangeable under the grid's symmetries and
no boundary is meaningfully more robust than another.

We then run a real `hierarchical_recom` chain for 400 steps,
discarding the first 40 (burn-in) and recording every other step
(thinning) — 180 samples in total.

```{code-cell} python
import json
from functools import partial

import networkx as nx

from falcomchain import (
    EnsembleStats,
    MarkovChain,
    Partition,
    always_accept,
    hierarchical_recom,
)
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed

with open("_static/demo_grid_10x10.json") as f:
    graph = nx.node_link_graph(json.load(f), edges="links")

# Manhattan-distance travel times keep the example tiny.
Assignment.travel_times = {
    (a, b): float(
        abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
        + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"])
    )
    for a in graph.nodes for b in graph.nodes
}

set_seed(42)

# Seed the initial partition on the *achievable* per-team workload.
# With total demand D and a workload cap of w per team, the coverage
# rule needs k = ceil(D / w) teams, so the k districts must average
# D / k, which is strictly below the nominal w. Centering the initial
# recursion on D / k (rather than on w) keeps the last district inside
# the balance window; this mirrors the recentering the chain proposal
# does locally. Capacities are kept in the safe range c in {1, 2}, where
# the recursive partitioner provably never stalls (see the paper); c >= 3
# needs the capacity-block reparametrization.
import math

demand_target = 1000
total_demand = sum(d["demand"] for _, d in graph.nodes(data=True))
k_teams = max(1, math.ceil(total_demand / demand_target))
seed_demand_target = total_demand / k_teams

partition = Partition.from_random_assignment(
    graph=graph,
    epsilon=0.3,
    demand_target=seed_demand_target,
    assignment_class=None,
    capacity_level=2,
)
state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)

ensemble = EnsembleStats(burn_in=40, thin=2)
chain = MarkovChain(
    proposal=partial(
        hierarchical_recom,
        epsilon_base=0.3,
        epsilon_super=0.3,
        demand_target=1000,
    ),
    constraints=lambda p: True,
    accept=always_accept,
    initial_state=state,
    total_steps=400,
    callbacks=[ensemble.observe],
)
list(chain)

ensemble.n_samples
```

## Inspecting boundary frequency

```{code-cell} python
boundary_freq = ensemble.boundary.frequencies()
print(f"Edges that ever appeared as boundaries: {len(boundary_freq)}")
print(f"Robust  (>=70% of samples): {len(ensemble.boundary.robust(0.7))}")
print(f"Fragile (<50% of samples):  {len(ensemble.boundary.fragile(0.5))}")
```

### Boundary heatmap

`falcomplot` ships the boundary-frequency map as a single call, so we do
not hand-roll the drawing. Import it as `fp` and pass the edge list, the
per-edge frequencies, and the node positions. Edges are colored by how
often they serve as a district boundary: dark = robust (nearly always a
boundary), pale = fragile.

```{code-cell} python
import falcomplot as fp

pos = {n: (d["C_X"], d["C_Y"]) for n, d in graph.nodes(data=True)}
edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
freqs = [boundary_freq.get(e, 0.0) for e in edges]

fig = fp.plot_boundary_frequency(
    edges, freqs, pos,
    aspect="equal", floor=0.0,
    colorbar_label="boundary frequency",
    title="Boundary frequency map (demo grid)",
)
```

The robust boundaries (dark) tend to lie *between* cities — these are
the cuts the chain makes consistently, regardless of starting state.
Boundaries within rural regions remain pale because the chain has
many equally good ways to draw them.

## Inspecting facility stability

```{code-cell} python
facility_freq = ensemble.facility.frequencies()
print(f"Candidates ever selected: {len(facility_freq)}")
print(f"Essential (>=90%):        {len(ensemble.facility.essential(0.9))}")
print(f"Substitutable (<50%):     {len(ensemble.facility.substitutable(0.5))}")
```

Selection rates read best as a chart — essential candidates (selected
in nearly every sample) stand apart from the substitutable tail:

```{code-cell} python
import matplotlib.pyplot as plt

ranked = sorted(facility_freq.items(), key=lambda kv: -kv[1])
nodes = [str(n) for n, _ in ranked]
rates = [f for _, f in ranked]

fig, ax = plt.subplots(figsize=(7, 3))
colors = ["#b3452c" if r >= 0.9 else "#33608c" if r >= 0.5 else "#a9bfd6"
          for r in rates]
ax.bar(range(len(nodes)), rates, color=colors)
ax.axhline(0.9, color="#b3452c", lw=0.8, ls="--")
ax.axhline(0.5, color="#a9bfd6", lw=0.8, ls="--")
ax.set_xticks(range(len(nodes)), nodes, rotation=90, fontsize=7)
ax.set_ylabel("selection frequency")
ax.set_title("Facility stability: essential (red) vs substitutable (pale)");
```

Candidates above the 0.9 line are *essential* — the chain picks them in
nearly every sampled plan, so removing them would degrade most good
designs. Candidates below 0.5 are *substitutable*: other sites cover
their role.

## Capacity utilization summary

Per-sample CV of demand-per-team and covering-radius statistics across
the ensemble:

```{code-cell} python
ensemble.capacity.summary()
```

## Full report

`report()` returns everything in one dict:

```{code-cell} python
report = ensemble.report()
sorted(report.keys())
```

```{code-cell} python
print(f"n_samples:                {report['n_samples']}")
print(f"essential_facilities:     {len(report['essential_facilities'])}")
print(f"substitutable_facilities: {len(report['substitutable_facilities'])}")
print(f"robust_boundaries:        {len(report['robust_boundaries'])}")
print(f"fragile_boundaries:       {len(report['fragile_boundaries'])}")
```

## Burn-in, thinning, and the callback hook

The chain example above already uses the standard MCMC accumulation
pattern: ``EnsembleStats(burn_in=40, thin=2)`` was passed as a
``callback`` to ``MarkovChain``, so the chain feeds every state into
``ensemble.observe`` as it runs. Out of 400 chain steps, the first
40 are discarded (burn-in) and only every other remaining step is
recorded (thinning), giving ``n_samples = 180``.

For longer chains, raise the burn-in and thinning to match — typical
real-world settings are ``burn_in=500-2000`` and ``thin=10`` for
chains of 10,000+ steps:

```python
ensemble = EnsembleStats(burn_in=500, thin=10)
chain = MarkovChain(..., total_steps=10_000, callbacks=[ensemble.observe])
list(chain)
report = ensemble.report()
```

## Filtering out artificial candidates

If you ran [`repair_facility_density`](feasibility.md) before the chain,
the ensemble counts artificial candidates alongside real ones. To
report only real essential facilities:

```python
real_essential = {
    n: f for n, f in report["essential_facilities"].items()
    if not graph.nodes[n].get("candidate_artificial", 0)
}
```

A high frequency on an *artificial* candidate is informative — it means
the chain wants a facility there — but should not be presented as
"essential" without that context.

## Convergence diagnostics

The ensemble statistics above are only meaningful once the chain has
reached its stationary regime. No closed-form stationary distribution is
known for recombination-style chains, FalCom included, so convergence is
assessed *empirically*, in the order the FalCom paper uses:

1. **Exact enumeration** on an instance small enough to list every
   feasible state: does the chain visit only feasible states, all of
   them, and with the same frequencies from very different starts?
2. **Start-independence on large instances**: do independently started
   chains agree on the distributions of summary statistics
   (Kolmogorov--Smirnov distance), and how fast does each chain forget
   its initial plan (share of the initial boundary edges still cut)?
3. **Gelman--Rubin $\hat R$ and effective sample size** as secondary
   numbers, computed by `falcomplot`.

### Exact enumeration on a 3x4 grid

The validation instance in the paper (and in the
[London-Ambulance-Service-System](https://github.com/kirtisoglu/London-Ambulance-Service-System)
repo, `falcomchain_experiments/validation/enumeration.py`) is a 3x4
grid of twelve units with demand 100, four candidate sites, `w = 300`,
`epsilon = 0.15`, `c1 in {1, 2}`, `c2 in [2, 4]` and `kappa = 2`, which
has 93 feasible level-1 partitions and 119 joint states. Three chains
of 200,000 steps from the first, middle and last partition visit all of
them and nothing else; the total-variation distance between any two
chains' empirical laws is at most 0.012, and between the two halves of
one chain at most 0.016.

```{figure} _static/fig_enumeration_3x4.png
:alt: exact enumeration validation
:width: 100%

(a) Sampled frequency of each feasible level-1 state against the
spanning-tree law; (b) probability mass by district-size pattern under
the sampled, uniform and spanning-tree laws.
```

The sampled law is neither uniform (total-variation distance 0.40) nor
the spanning-tree law (0.37). Across district-size patterns it spreads
its mass almost like the uniform law; within a pattern the frequencies
track the spanning-tree weights (log-log correlation 0.94), the same
compactness preference ReCom has. The spanning-tree law conditioned on
the pattern is within 0.11 of the sampled law. Read the ensemble
statistics of this page as readouts of *that* law.

### Start-independence with FalcomPlot

Record a scalar summary along each chain — here the number of cut
(boundary) edges, though the energy or a district count works just as
well — then hand the list of traces to `falcomplot`. We run three
independently seeded chains on the same demo grid:

```{code-cell} python
import falcomplot as fp

def run_chain(seed, steps=200):
    set_seed(seed)
    partition = Partition.from_random_assignment(
        graph=graph, epsilon=0.3, demand_target=seed_demand_target,
        assignment_class=None, capacity_level=2)
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    trace = []
    def record(st, accepted):
        m = st.partition.assignment.mapping
        trace.append(sum(1 for u, v in graph.edges() if m[u] != m[v]))
    chain = MarkovChain(
        proposal=partial(hierarchical_recom, epsilon_base=0.3,
                         epsilon_super=0.3, demand_target=demand_target),
        constraints=lambda p: True, accept=always_accept,
        initial_state=state, total_steps=steps, callbacks=[record])
    list(chain)
    return trace

burn_in = 20
traces = [run_chain(seed) for seed in (1, 2, 3)]

rhat = fp.gelman_rubin([t[burn_in:] for t in traces])
ess  = sum(fp.effective_sample_size(t[burn_in:]) for t in traces)
print(f"R-hat = {rhat:.3f}   ESS = {ess:.0f}")
```

`plot_convergence` returns a two-panel figure: the left panel overlays
the chains' traces with the burn-in shaded, and the right panel shows
their post-burn-in distributions with the split $\hat R$ annotated:

```{code-cell} python
fig = fp.plot_convergence(
    traces, burn_in=burn_in, value_label="cut edges",
    labels=[f"chain {i}" for i in (1, 2, 3)],
)
```

A single-chain trace is drawn with `fp.plot_trace`, and a
boundary-frequency map over a real dual graph with
`fp.plot_boundary_frequency`, exactly as above. Chains that agree on
their summary statistics can still share a region they never leave;
that is why the enumeration check above and the forgetting curves come
first, and $\hat R$ last.

### In the FalCom paper: the London Ambulance Service ensemble

On the London Ambulance Service instance (a 4,994-node LSOA dual graph
whose only candidates are the 66 real stations), four chains of 40,000
steps were run from independent sector-wise initial plans at the locked
capacity-block calibration. Acceptance is 38--39%, with
almost every rejection at the supergraph recursion (`chain.rejection_report()`
counts them by cause). After a burn-in of 8,000 steps the four chains agree on
the energy (largest pairwise KS distance 0.048, split $\hat R = 1.003$),
on the number of districts (mean 53, $\hat R = 1.006$) and on the number of
super-districts (mean 22.6, $\hat R = 1.001$); the share of each chain's
initial boundary still in place falls to the independent-plan level of 0.20
within about 1,000 steps. The ensemble opens 46--61 of the 66 stations per
plan, keeps 21 of them in more than 90% of plans, and no level-1 boundary
edge is cut in more than 47% of plans.

```{figure} _static/las_traces.png
:alt: London Ambulance real-station chains
:width: 100%

Energy, district-count and cut-edge traces of the four real-station
chains (left, burn-in shaded) and their post-burn-in distributions
(right).
```

```{figure} _static/las_boundary_freq.png
:alt: London Ambulance boundary frequency
:width: 100%

Level-1 district and level-2 super-district boundary frequencies across
the real-station ensemble.
```

See the FalCom paper for the full setup and metrics.

## API reference

### `EnsembleStats(burn_in=0, thin=1)`

Main coordinator. Holds `boundary`, `facility`, and `capacity` analyzers.

- `observe(state, accepted)` — record one step
- `report()` — return summary dict
- `n_samples` — number of recorded samples

### `BoundaryCounter()`

- `observe(state, accepted)`
- `frequencies()` — dict[edge, float]
- `robust(threshold=0.9)` — edges above threshold
- `fragile(threshold=0.5)` — edges below threshold

### `FacilityCounter()`

- `observe(state, accepted)`
- `frequencies()` — dict[node, float] (level-1)
- `super_frequencies()` — dict[node, float] (level-2)
- `essential(threshold=0.9)` — nodes above threshold
- `substitutable(threshold=0.5)` — nodes below threshold

### `CapacityStats()`

- `observe(state, accepted)`
- `summary()` — dict with `demand_cv`, `max_radius`, `mean_radius` (each a Welford summary)
- `demand_cv`, `max_radius`, `mean_radius` — `WelfordStats` instances

### `WelfordStats()`

Online mean/variance via Welford's algorithm.

- `update(x)` — add observation
- `mean`, `variance`, `std`, `min`, `max`, `n` — properties
- `summary()` — dict with all stats
