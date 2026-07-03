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

# Level-2 (Super-) Facilities

FalcomChain supports two-level hierarchical facility location: each
*superdistrict* (a group of level-1 districts) can host a level-2
facility chosen from its own candidate set `F²`. Level-2 facility
assignment is **opt-in** and **soft**: you only compute it if you ask
for it, and superdistricts with no eligible candidate simply get no
level-2 facility.

This page walks through the level-2 workflow on the shared demo grid.

## The candidate sets are independent

Eq. 18 in the paper defines `F² ⊂ V¹` — level-2 candidates are nodes of
the base graph, **not** a subset of the level-1 facility set `F¹`. They
are an independent property of nodes:

- `candidate=1` → eligible to host a level-1 facility
- `super_candidate=1` → eligible to host a level-2 facility

A node can be either, both, or neither. The shared demo grid has
about 30 level-1 candidates and 5 level-2 candidates, with several
overlapping — the visualization makes the independence clear.

```{code-cell} python
import json
import networkx as nx

with open("_static/demo_grid_10x10.json") as f:
    data = json.load(f)
graph = nx.node_link_graph(data, edges="links")

n_l1 = sum(1 for n, d in graph.nodes(data=True) if d.get("candidate"))
n_l2 = sum(1 for n, d in graph.nodes(data=True) if d.get("super_candidate"))
n_both = sum(
    1 for n, d in graph.nodes(data=True)
    if d.get("candidate") and d.get("super_candidate")
)
print(f"level-1 candidates:   {n_l1}")
print(f"level-2 candidates:   {n_l2}")
print(f"overlap (both):       {n_both}")
```

## Plotting both candidate layers

Two markers stacked: blue circles for level-1 candidates, red squares
for level-2 candidates. Overlapping nodes show both. We use
:func:`falcomplot.plot_grid` from the
[FalcomPlot](https://github.com/kirtisoglu/FalcomPlot) companion
library — same helper used across the FalcomChain docs.

```{code-cell} python
from falcomplot import plot_grid

plot_grid(graph, title="Demo grid — level-1 vs level-2 candidate sets");
```

## Opting into level-2 facility assignment

By default, `state.super_facility` is `None`. To enable Eq. 18, pass
`super_facility_fn=SuperFacilityAssignment.from_state` to
`ChainState.initial`:

```{code-cell} python
from falcomchain import Partition, SuperFacilityAssignment
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed

set_seed(42)

# Manhattan-distance travel times on the grid.
Assignment.travel_times = {
    (a, b): float(abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
                  + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"]))
    for a in graph.nodes for b in graph.nodes
}

# Seed on the achievable per-team load total / k (k = ceil(total / w)),
# not the raw target w; keep capacities in the safe range c in {1, 2}.
import math
_total = sum(d["demand"] for _, d in graph.nodes(data=True))
seed_demand_target = _total / max(1, math.ceil(_total / 1000))

partition = Partition.from_random_assignment(
    graph=graph,
    epsilon=0.3,
    demand_target=seed_demand_target,
    assignment_class=None,
    capacity_level=2,
)

state = ChainState.initial(
    partition=partition,
    energy=0.0,
    beta=1.0,
    super_facility_fn=SuperFacilityAssignment.from_state,
)
state.super_facility
```

## Inspecting the level-2 assignment

In the **initial** partition, the super_assignment is identity — each
level-1 district is its own superdistrict. So the level-2 facility for
superdistrict `s` is just the best super-candidate inside that one
level-1 district.

```{code-cell} python
print(f"Number of level-1 districts:        {len(partition.parts)}")
print(f"Number of superdistricts:           {len(partition.super_parts)}")
print(f"Superdistricts with a level-2 fac.: {len(state.super_facility.centers)}")
print(f"Superdistricts without (soft skip): "
      f"{len(partition.super_parts) - len(state.super_facility.centers)}")
```

```{code-cell} python
plot_grid(
    graph,
    title="Initial partition — level-2 facilities (black rings)",
    super_centers=state.super_facility.centers.values(),
);
```

The black rings mark superdistricts where a level-2 facility was
selected (a super-candidate exists in that superdistrict). Other
superdistricts have no level-2 facility — that's the soft constraint
in action.

## After a chain step the picture changes

After `hierarchical_recom`, the supergraph is repartitioned and some
level-1 districts get grouped into multi-district superdistricts. Each
group now has more base nodes to consider, so the level-2 selection
can place a facility serving a wider area.

```{code-cell} python
from falcomchain import hierarchical_recom

set_seed(7)
new_state = hierarchical_recom(
    state, epsilon_base=0.3, epsilon_super=0.3, demand_target=1000,
)
print(f"new superdistricts:          {len(new_state.partition.super_parts)}")
print(f"with a level-2 facility:     {len(new_state.super_facility.centers)}")
print(f"sample super_parts (first 3):")
for sid, l1_ids in list(new_state.partition.super_parts.items())[:3]:
    print(f"  super_id {sid}: level-1 districts {sorted(l1_ids)}")
```

## A custom selector

`SuperFacilityAssignment.from_state` takes a `selection_fn` callable.
The default is :func:`minimax_super_selector` (Eq. 18). You can plug in
any function with the same signature — e.g., demand-weighted minimax,
or a tie-breaker that prefers certain candidates.

```python
from falcomchain.markovchain.facility import minimax_super_selector

def demand_weighted_super_selector(super_id, super_candidates, base_nodes,
                                   travel_times):
    \"\"\"Pick c minimizing max(demand[v] * dist(c, v)) over base_nodes.\"\"\"
    # ... your logic ...

state = ChainState.initial(
    partition=partition,
    energy=0.0,
    beta=1.0,
    super_facility_fn=lambda s: SuperFacilityAssignment.from_state(
        s, selection_fn=demand_weighted_super_selector
    ),
)
```

## Biasing the supergraph cut toward good level-2 candidates

The level-2 facility *assignment* (above) runs after the supergraph
partition has been chosen. To also bias the *cut selection itself*
toward subtrees that admit well-located level-2 candidates, pass
``gamma_super > 0`` to :func:`hierarchical_recom`. With γ² > 0 the
chain uses the paper's hub-coherence score (Eq. 22, 27):

```
ψ²(T_u) = ϕ²(T_u) · exp(-γ² · π²(T_u))
π²(T_u) = min_{f ∈ F² ∩ V¹[T_u]} max_{v ∈ V(T_u)} d(f, f¹(D_v¹))
```

For each super-candidate `f` in the subtree, the inner max is the
worst-case distance from `f` to any level-1 facility center in `T_u`.
The outer min picks the candidate with smallest such eccentricity.
Subtrees with no super-candidate get π² = +∞ and are filtered out
(soft level-2 constraint, paper Section 5.5).

```python
from falcomchain import hierarchical_recom
from functools import partial

proposal = partial(
    hierarchical_recom,
    epsilon_base=0.1,
    epsilon_super=0.15,
    demand_target=1500,
    gamma_super=1.0,    # 0 = uniform; large = greedy on hub-coherence
)
```

For a custom level-2 cut score, pass `super_psi_fn` directly — a
callable `(subnodes, teams) -> float`. See
:func:`hub_coherence_psi_factory` for the default implementation
pattern.

## Using `super_facility` in ensemble analysis

If `super_facility_fn` is set, `state.super_facility.centers` is part
of every sample. The ensemble's `super_facility_frequencies` (in the
report from :class:`EnsembleStats`) counts how often each
super-candidate is selected across the chain — the level-2 analog of
the level-1 facility stability metric.

When `super_facility_fn=None` (default), `state.super_facility` is
`None`, and the ensemble's `super_facility_frequencies` stays empty.

## API reference

### `SuperFacilityAssignment.from_state(state, selection_fn=None)`

Compute level-2 facilities for every superdistrict in
`state.partition.super_parts`. Soft-skips superdistricts whose
constituent base nodes contain no super-candidate.

- `selection_fn` — callable
  `(super_id, super_candidates, base_nodes, travel_times) -> (node, radius)`.
  Default: :func:`minimax_super_selector`.

### `minimax_super_selector(super_id, super_candidates, base_nodes, travel_times)`

The default Eq. 18 minimax: pick the super-candidate minimising the
maximum travel time to any base node in the superdistrict. Returns
`(best_candidate, covering_radius)`.

### `ChainState.initial(..., super_facility_fn=None)`

`super_facility_fn=None` (default) ⇒ `state.super_facility is None`
for the whole chain. Pass `SuperFacilityAssignment.from_state` (or any
custom callable taking a `ChainState` and returning a
`SuperFacilityAssignment`) to enable.
