# Getting Started

FalcomChain is a Python library for hierarchical capacitated facility location
via Markov chain Monte Carlo (MCMC). This guide walks you through your first
chain in five minutes.

## Installation

```bash
pip install falcomchain
```

Requires Python 3.12+.

## The big picture

A FalcomChain workflow has four steps:

1. **Build a `Graph`** from your geographic data (shapefile, GeoJSON, raw dicts).
2. **Create an initial `Partition`** of nodes into districts.
3. **Run a `MarkovChain`** for N steps.
4. **Analyze** the resulting ensemble (boundary frequency, facility stability, etc.).

```python
from falcomchain import Graph, Partition, MarkovChain
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.accept import always_accept
from functools import partial

# 1. Build graph
graph = Graph.from_file("districts.shp", demand_col="POP", candidate_col="is_clinic")

# 2. Initial partition
partition = Partition.from_random_assignment(
    graph=graph,
    epsilon=0.1,
    demand_target=1500,
    assignment_class=None,
    capacity_level=3,
)

# 3. Run the chain
proposal = partial(hierarchical_recom, epsilon=0.1, demand_target=1500)
chain = MarkovChain(
    proposal=proposal,
    constraints=lambda p: True,
    accept=always_accept,
    initial_state=initial_state,  # see "Travel times" below
    total_steps=1000,
)

for state in chain:
    pass  # collect samples
```

## Required graph attributes

Your graph needs two node attributes for the algorithms to work:

| Attribute | Type | Purpose |
|---|---|---|
| `demand` | float | Demand of the unit (population, workload) |
| `candidate` | int (0/1) | Whether the node can host a facility |

Two more are recommended:

| Attribute | Type | Purpose |
|---|---|---|
| `C_X`, `C_Y` | float | Centroid coordinates (for visualization) |
| `area` | float | Geographic area (for compactness metrics) |

See [schema reference](schema.md) for the full schema.

## Travel times

The energy function and facility assignment use a travel-time matrix. Set it
once before running the chain:

```python
from falcomchain.partition.assignment import Assignment

travel_times = {(facility_node, node): float for ...}
Assignment.travel_times = travel_times
```

For testing, use Manhattan or Euclidean distance from the coordinates.

## Saving and loading

A graph with a partition baked in is a complete chain state:

```python
# Save
partition.write_to_graph()  # writes 'district' attr to nodes
import pickle
with open("state.pkl", "wb") as f:
    pickle.dump(graph, f)

# Load
with open("state.pkl", "rb") as f:
    graph = pickle.load(f)
partition = Partition.from_graph(graph)
```

## Visualizing with FalcomPlot

To produce data for the FalcomPlot animation:

```python
from falcomchain.tree.snapshot import Recorder

recorder = Recorder("output/", record_substeps=True)
recorder.write_header(graph, partition, params={"epsilon": 0.1, ...})

chain = MarkovChain(..., recorder=recorder)
for state in chain:
    pass
recorder.close()

# Export to JSON for the dashboard
Recorder.export_to_json("output/", "output/json/")
```

See [GeoDataFrame guide](geodataframe.md) for working with shapefiles.
