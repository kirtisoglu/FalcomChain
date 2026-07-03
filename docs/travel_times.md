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

# Travel Times with FalcomTravel

```{admonition} Goal of this page
:class: tip
Produce the travel-time matrix that FalcomChain consumes — with
[FalcomTravel](https://github.com/kirtisoglu/FalcomTravel) for real
road networks, or with a two-line fallback for prototyping.
```

Travel times drive two things in FalcomChain:

1. **Facility assignment** — the minimax facility center of each district
   (Eq. 17) is the candidate minimizing the worst travel time to any node.
2. **The optional energy objective** — `compute_energy` is demand-weighted
   travel time (see [Optimization Methods](optimization_methods.md)).

FalcomChain does not compute travel times itself. It reads one class
attribute:

```python
from falcomchain.partition.assignment import Assignment

Assignment.travel_times = {...}   # {(origin, destination): seconds}
```

Any dict-like `(u, v) -> float` works, as long as the keys match
`graph.nodes` exactly. This page shows how to fill it.

## The contract

| Property | Requirement |
|---|---|
| Keys | `(origin_node, destination_node)` tuples using the **same node IDs as your graph** |
| Values | non-negative floats (seconds by convention) |
| Coverage | every `(candidate, node)` pair the chain may evaluate |
| Symmetry | not required — real road networks are asymmetric |

```{warning}
The most common real-data failure is a key mismatch: the routing engine
returns GEOID strings while the graph uses integer node IDs. Build a key
map before assignment — see the checklist at the end of this page.
```

## FalcomTravel in one look

FalcomTravel is the FalCom family's travel-time library: a thin,
multi-backend façade over existing routing engines that returns exactly
the dict shape above. One output contract, one `Mode` enum
(`DRIVE`, `WALK`, `BIKE`, `TRANSIT`, `DRIVE_TRANSIT`), translated
per backend.

| Backend | Best for | External deps |
|---|---|---|
| `OSMnxBackend` | Driving / walking / biking, no toolchain | `osmnx`, `scipy` (pip) |
| `R5RBackend` | Multimodal urban routing with GTFS | R, Java, `r5r` R package |
| `GraphBackend` | Synthetic grids, custom weighted graphs | none |
| `EuclideanBackend` | Sanity baselines, lower bounds | none |

```bash
pip install "falcomtravel @ git+https://github.com/kirtisoglu/FalcomTravel.git@main"
```

## Example: travel times for the demo grid

The docs' shared 10×10 demo grid has no road network, so we use the
two dependency-free backends. `GraphBackend` runs Dijkstra on any
weighted NetworkX graph:

```{code-cell} python
import json
import networkx as nx
import falcomtravel as ft

with open("_static/demo_grid_10x10.json") as f:
    graph = nx.node_link_graph(json.load(f), edges="links")

# Give every edge a traversal time (here: 60 s per grid step).
for u, v in graph.edges:
    graph.edges[u, v]["time"] = 60.0

backend = ft.GraphBackend(graph, weight="time")
matrix = backend.compute(
    origins=list(graph.nodes),
    destinations=list(graph.nodes),
    mode=ft.Mode.DRIVE,   # informational for GraphBackend; weights decide
)
print(f"{len(matrix.as_dict())} OD pairs")
```

Hand the result to FalcomChain:

```{code-cell} python
from falcomchain.partition.assignment import Assignment

Assignment.travel_times = matrix.as_dict()
sample = next(iter(Assignment.travel_times.items()))
print(f"sample entry: {sample}")
```

That's it — every page in these docs that runs a chain does exactly
this step (most of them with a hand-rolled Manhattan dict, which is the
same idea without the dependency).

`EuclideanBackend` gives straight-line-distance baselines from node
coordinates. On abstract grids pass `geodesic=False` (the default
assumes lon/lat):

```{code-cell} python
coords = ft.coords_from_graph(graph, x="C_X", y="C_Y")
baseline = ft.EuclideanBackend(coords, geodesic=False).compute(
    origins=list(graph.nodes),
    destinations=list(graph.nodes),
    mode=ft.Mode.DRIVE,
)
d_graph = matrix.get((0, 99))
d_line  = baseline.get((0, 99))
print(f"corner to corner: graph {d_graph:.0f}s vs straight-line {d_line:.4f}s")
```

Straight-line values are in different units (distance-derived), so use
them for *relative* sanity checks — is the graph metric roughly
monotone in the Euclidean one? — not as drop-in travel times.

## Pre-flight diagnostics

Real OSM extracts routinely contain isolated road fragments. A node on
such a fragment silently produces missing OD pairs an hour into an r5r
run. `diagnose()` catches this before you route:

```{code-cell} python
report = ft.diagnose(graph, origins=list(graph.nodes)[:5],
                     destinations=graph.nodes)
print(f"ok: {report.ok}")
```

On a real Chicago driving network the same call finds two census blocks
on isolated fragments — the source of 5,496 missing OD pairs that r5r
would have dropped without a word. If `report.ok` is false, print
`report.summary()` and decide: drop the disconnected nodes, snap them to
the nearest reachable component, or re-extract OSM with a wider boundary.

## Real road networks

For actual driving/walking/biking times, `OSMnxBackend` fetches the OSM
network on the fly and runs multi-source Dijkstra in C via
`scipy.sparse.csgraph` — no R, no Java:

```python
import falcomtravel as ft

coords = ft.coords_from_graph(graph, x="C_X", y="C_Y")  # (lon, lat) per node

backend = ft.OSMnxBackend.from_place(
    "Chicago, Illinois, USA",
    coords=coords,
    network_type="drive",
)
matrix = backend.compute(
    origins=candidate_nodes,       # facility candidates
    destinations=list(coords),     # everything
    mode=ft.Mode.DRIVE,
)
Assignment.travel_times = matrix.as_dict()
```

When you need transit and GTFS-aware routing, use `R5RBackend`
(requires R 4.0+, Java 21+, and the `r5r` R package):

```python
backend = ft.R5RBackend(
    data_path="data/london",        # OSM .pbf + GTFS .zip
    coords=coords,
    timezone="Europe/London",
)
matrix = backend.compute(
    origins=candidate_nodes,
    destinations=list(coords),
    mode=ft.Mode.TRANSIT,
    departure_time="2025-09-08 09:00:00",
    max_trip_duration=900,          # minutes
)
```

## Persisting large matrices

A 5,000-node instance has up to 25 M OD pairs. Compute once, persist,
reload:

```python
matrix.to_parquet("output/ttm.parquet")

# Later — or directly from r5r's partitioned write_dataset output:
matrix = ft.TravelMatrix.from_parquet("output/ttm.parquet", cast=int)
Assignment.travel_times = matrix.as_dict()
```

## Prototyping without FalcomTravel

For synthetic experiments, a dict comprehension over node coordinates
is all you need — this is what the other doc pages do:

```python
Assignment.travel_times = {
    (a, b): float(
        abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
        + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"])
    )
    for a in graph.nodes for b in graph.nodes
}
```

## Checklist before running a chain

1. **Key types match** — `set(matrix.as_dict())` keys use the same IDs
   (and the same *types*: `int` vs `str`) as `graph.nodes`.
2. **Candidates covered** — every `(candidate, node)` pair within a
   plausible district is present.
3. **No negative or NaN values** — clean the engine output first
   (`matrix.fill_missing(...)` helps).
4. **Ran `diagnose()`** on the routing network if it came from OSM.

## Where to go next

- [Running a FalCom Chain](running_a_chain.md) — use the matrix in a chain.
- [Case Study: London Ambulance Service](working_with_real_data.md) — a
  real instance where travel times come from a road network.
- [Optimization Methods](optimization_methods.md) — the energy objective
  that consumes travel times.
