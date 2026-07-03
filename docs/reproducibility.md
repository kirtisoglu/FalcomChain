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

# Reproducibility

```{admonition} Goal of this page
:class: tip
Make any chain run exactly repeatable — what `set_seed` guarantees,
what else must be held fixed, and how to package a run so someone else
(including reviewers) can regenerate your results bit for bit.
```

MCMC results are only credible if they can be regenerated. FalcomChain
is designed so that **the same seed, the same inputs, and the same
library versions produce the identical chain** — every proposal, every
cut, every facility assignment.

## The isolated RNG

All stochastic operations in the library draw from a dedicated
generator, `falcomchain.random.rng`, never from Python's global
`random` module. This isolation matters: a third-party library calling
`random.random()` somewhere inside your pipeline (geopandas, networkx,
plotting code) cannot perturb the chain's random state.

```python
from falcomchain.random import set_seed

set_seed(2025)   # deterministic from here on
```

Call `set_seed` **before constructing the initial partition** — the
initial `Partition.from_random_assignment` consumes random draws, so
seeding after it reproduces the chain but not the starting state.

## Determinism, demonstrated

Two runs with the same seed, compared cut by cut. This cell executes at
docs build time, so what you see below is a live check, not a promise:

```{code-cell} python
import json
import math
from functools import partial

import networkx as nx

from falcomchain import (
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

Assignment.travel_times = {
    (a, b): float(
        abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
        + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"])
    )
    for a in graph.nodes for b in graph.nodes
}


def run_chain(seed, steps=100):
    set_seed(seed)                      # seed BEFORE the initial partition
    total = sum(d["demand"] for _, d in graph.nodes(data=True))
    seed_target = total / max(1, math.ceil(total / 1000))
    partition = Partition.from_random_assignment(
        graph=graph, epsilon=0.3, demand_target=seed_target,
        assignment_class=None, capacity_level=2,
    )
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    chain = MarkovChain(
        proposal=partial(hierarchical_recom, epsilon_base=0.3,
                         epsilon_super=0.3, demand_target=1000),
        constraints=lambda p: True, accept=always_accept,
        initial_state=state, total_steps=steps,
    )
    trace, final = [], None
    for s in chain:
        trace.append(len(s.partition.parts))
        final = s
    return trace, dict(final.partition.assignment.mapping), dict(final.facility.centers)


trace_a, assign_a, centers_a = run_chain(seed=42)
trace_b, assign_b, centers_b = run_chain(seed=42)
trace_c, assign_c, centers_c = run_chain(seed=7)

print(f"same seed  → identical district-count trace: {trace_a == trace_b}")
print(f"same seed  → identical final assignment:     {assign_a == assign_b}")
print(f"same seed  → identical facility centers:     {centers_a == centers_b}")
print(f"other seed → different chain:                {assign_a != assign_c}")
```

## What the seed does *not* cover

`set_seed` fixes the chain's randomness. Full reproducibility also
requires that everything the chain *reads* is identical:

| Input | Requirement |
|---|---|
| **Graph** | Same node set, attributes, and edges. Serialize it (pickle or `save_json`) rather than rebuilding from geodata each time — geometry cleaning and adjacency computation should not be repeated per run. |
| **Travel times** | `Assignment.travel_times` must be the identical mapping. Persist the matrix ([`matrix.to_parquet`](travel_times.md)) and reload it. |
| **Parameters** | `epsilon`, `demand_target`, `capacity_level`, `gamma`, acceptance rule, `total_steps`, burn-in/thinning. Record them next to the seed. |
| **Library versions** | Same `falcomchain` version (an algorithm change can alter the draw sequence even with the same seed). Pin with `pip freeze` or a lock file. |
| **Python version** | Python's `random.Random` is stable across versions in practice, but record it anyway (3.12+ is required). |

A convenient pattern is a single run manifest:

```python
manifest = {
    "seed": 42,
    "total_steps": 10_000,
    "epsilon": 0.1, "demand_target": 1500, "capacity_level": 3,
    "gamma": 1.0, "accept": "always_accept",
    "burn_in": 500, "thin": 10,
    "falcomchain": importlib.metadata.version("falcomchain"),
    "graph_file": "graph.pkl", "travel_times_file": "ttm.parquet",
}
json.dump(manifest, open("run_manifest.json", "w"), indent=2)
```

## Independent chains need independent seeds

For convergence diagnostics ($\hat R$, ESS — see
[Ensemble Analysis](ensemble.md)) you run several chains that must be
*independent but individually reproducible*. Use distinct, recorded
seeds — not `set_seed` once for all chains:

```python
traces = [run_chain(seed=s) for s in (1, 2, 3, 4)]   # record (1, 2, 3, 4)
```

## Reproducing a *state* vs reproducing a *run*

You don't always need to re-run the chain. A graph with the partition
written into it is a complete, portable description of a chain state:

```python
partition.write_to_graph()                 # 'district' attr + team counts
with open("state.pkl", "wb") as f:
    pickle.dump(graph, f)

# elsewhere / later
graph = pickle.load(open("state.pkl", "rb"))
partition = Partition.from_graph(graph)
```

And for step-by-step forensics of a past run, attach a
[`Recorder`](visualization.md) during the run — the binary trajectory
(`chain.fcrec`) replays every proposal, spanning tree, and cut without
re-executing anything.

## Checklist for a reproducible experiment

1. `set_seed(s)` before the initial partition; record `s`.
2. Serialize the graph and travel-time matrix; load, don't rebuild.
3. Save a run manifest (parameters + seed + versions) next to outputs.
4. One distinct recorded seed per independent chain.
5. Pin library versions (`pip freeze > requirements.lock`).
6. For paper figures, also save the ensemble outputs
   (`EnsembleStats.report()`) so plots can be regenerated without
   re-sampling.

## Where to go next

- [Running a FalCom Chain](running_a_chain.md) — the run being made
  reproducible here.
- [Ensemble Analysis](ensemble.md) — multi-chain diagnostics that rely
  on independent seeds.
- [Contributing](contributing.md) — bug reports should include the
  seed; this page is why that suffices.
