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

# Fixed Superdistricts

```{admonition} Goal of this page
:class: tip
An advanced variant: hold an externally given level-2 partition (health
zones, sales territories) fixed while the chain explores level-1
districtings within each zone. Most users want the default resampling
mode instead.
```

Real-world hierarchical problems often come with the upper-level
partition already given — health zones, sales territories, school
districts. In those cases, you don't want the chain to resample the
level-2 partition every step; you want it to **explore level-1
districtings within each fixed zone**.

FalcomChain supports this via:

1. `Partition.from_random_assignment(super_assignment=...)` — build the
   initial partition by partitioning each zone independently.
2. `hierarchical_recom(super_partitioner=fixed_super_partition)` — run
   the chain with the level-2 partition held fixed.

## Setting up the zones

This page uses a small **uniform-demand** 10×6 synthetic grid, not the
shared heterogeneous demo grid. Reason: the demo grid has 5 high-demand
"city centers" that make balanced zone-based partitioning very hard
(bipartition_tree fails to find admissible cuts). With uniform demand
the per-zone arithmetic is simple and the demo is illustrative.

We split the grid into a left zone (`A`) and a right zone (`B`) at
`x = 5`.

```{code-cell} python
from falcomchain.graph.grid import Grid
from falcomchain.random import set_seed

set_seed(42)
graph = Grid(
    dimensions=(10, 6), num_candidates=20, density="uniform",
).graph

zones = {
    n: ("A" if n[0] < 5 else "B")
    for n in graph.nodes
}
print(f"zone A: {sum(1 for v in zones.values() if v == 'A')} nodes")
print(f"zone B: {sum(1 for v in zones.values() if v == 'B')} nodes")
```

## Building the initial partition

```{code-cell} python
from falcomchain import Partition
from falcomchain.partition.assignment import Assignment

set_seed(42)
Assignment.travel_times = None  # not needed for this demo

# Seed on the achievable per-team load total / k (k = ceil(total / w)),
# not the raw target w; keep capacities in the safe range c in {1, 2}.
import math
_total = sum(d["demand"] for _, d in graph.nodes(data=True))
seed_demand_target = _total / max(1, math.ceil(_total / 500))

partition = Partition.from_random_assignment(
    graph=graph,
    epsilon=0.3,
    demand_target=seed_demand_target,
    assignment_class=None,
    capacity_level=2,
    super_assignment=zones,
)

print(f"superdistricts:    {sorted(partition.super_parts.keys())}")
print(f"level-1 districts: {len(partition.parts)}")
for super_id, l1_ids in partition.super_parts.items():
    print(f"  zone {super_id!r}: {len(l1_ids)} level-1 district(s)")
```

Every level-1 district lies entirely inside one zone — the partition
respects the zone boundaries by construction.

## Visualizing the fixed zones

We pass per-node colors to :func:`falcomplot.plot_grid` — base nodes
are tinted by zone, and level-1 candidates are overlaid as blue
circles by the helper.

```{code-cell} python
from falcomplot import plot_grid

zone_colors = {"A": "#a6cee3", "B": "#fdbf6f"}
node_to_l1 = partition.assignment.mapping
node_color_map = {
    n: zone_colors[partition.super_assignment[node_to_l1[n]]]
    for n in graph.nodes
}

plot_grid(
    graph,
    title="Initial partition: two fixed zones",
    node_colors=node_color_map,
    show_level2_candidates=False,
);
```

## Running the chain with fixed superdistricts

`fixed_super_partition` keeps `partition.super_assignment` constant
across steps. Each step picks one zone uniformly and re-partitions the
level-1 districts within it.

```{code-cell} python
from falcomchain import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.super_partitioners import fixed_super_partition

# travel_times is required for the level-1 facility minimax (Eq. 17).
Assignment.travel_times = {
    (a, b): float(
        abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
        + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"])
    )
    for a in graph.nodes for b in graph.nodes
}

state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
print("Before any step:")
print(f"  super_parts keys: {sorted(state.partition.super_parts.keys())}")

set_seed(7)
new_state = hierarchical_recom(
    state,
    epsilon_base=0.3,
    epsilon_super=0.3,
    demand_target=500,
    super_partitioner=fixed_super_partition,
)
print("After one fixed-superdistrict step:")
print(f"  super_parts keys: {sorted(new_state.partition.super_parts.keys())}")
print(f"  level-1 districts: {len(new_state.partition.parts)}")
```

Same set of zones, level-1 districts have been resampled within
whichever zone was picked.

## Contrast: resample mode changes super_parts

For comparison, the default `resample_super_partition` builds a fresh
level-2 partition each step. With this small graph the resampled
partition typically has the same number of superdistricts as level-1
districts (each level-1 district becomes its own superdistrict), so
the result is materially different from the fixed-zone case.

```{code-cell} python
from falcomchain.markovchain.super_partitioners import resample_super_partition

# Use a non-zone partition to compare apples to apples.
set_seed(42)
plain_partition = Partition.from_random_assignment(
    graph=graph, epsilon=0.3, demand_target=seed_demand_target,
    assignment_class=None, capacity_level=2,
)
plain_state = ChainState.initial(
    partition=plain_partition, energy=0.0, beta=1.0
)

set_seed(8)
resampled_state = hierarchical_recom(
    plain_state,
    epsilon_base=0.3,
    epsilon_super=0.3,
    demand_target=500,
    super_partitioner=resample_super_partition,
)
print(f"Initial super_parts:    {len(plain_state.partition.super_parts)}")
print(f"After resample step:    {len(resampled_state.partition.super_parts)}")
```

## When to use fixed superdistricts

Use `fixed_super_partition` when:

- Superdistricts are policy-defined (zones, regions, watersheds).
- You want the ensemble to characterize *level-1* uncertainty within
  each fixed zone, not joint uncertainty over both levels.
- The level-2 facility ensemble is meaningful only relative to the
  given zones.

Use `resample_super_partition` (default) when:

- The hierarchical structure itself is part of what you're sampling.
- You want a free joint ensemble over level-1 *and* level-2 partitions.

## A note on ensemble analysis under fixed superdistricts

When the level-2 partition is fixed, the boundaries between zones
appear as boundaries in **every** sample — so they have boundary
frequency `1.0` and look "robust" in :class:`EnsembleStats.boundary`.
That's correct but not informative: those boundaries are policy
inputs, not chain findings. When reporting robust boundaries, filter
out edges whose endpoints lie in different zones:

```python
inter_zone = {
    (u, v) for u, v in graph.edges()
    if zones[u] != zones[v]
}
robust_within_zone = {
    e: f for e, f in ensemble.boundary.robust(0.9).items()
    if e not in inter_zone and (e[1], e[0]) not in inter_zone
}
```

## API reference

### `Partition.from_random_assignment(..., super_assignment=None)`

When `super_assignment` is provided, it must be a dict mapping every
base node to a superdistrict ID. The function partitions each zone
independently and stitches the results, with renumbered level-1 IDs
to avoid collisions across zones. The returned partition has its
`super_assignment` set to the user-provided zoning (mapping level-1
IDs to zone IDs).

### `fixed_super_partition(state, epsilon, demand_target, density=None, recorder=None)`

Picks one superdistrict uniformly at random from
`state.partition.super_parts`, returns the level-1 IDs in it as the
`merge` set, and carries the parent's `super_assignment` forward
unchanged. Raises `ValueError` if `super_parts` is empty.
