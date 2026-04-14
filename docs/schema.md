# Graph Attribute Schema

FalcomChain reads and writes specific attributes on graphs. This page is the
authoritative reference. The schema is defined in
[`falcomchain/graph/schema.py`](../falcomchain/graph/schema.py) and validated
at construction time.

You can print the schema at any time:

```python
from falcomchain.graph import describe_schema
print(describe_schema())
```

## Node attributes

These live on each node: `graph.nodes[node][attr]`.

### Required

| Name | Type | Default | Purpose |
|---|---|---|---|
| `demand` | float | — | Demand of the unit (population, workload, requests/day). Used for demand-balance constraints. Must be `>= 0`. |
| `candidate` | int (0 or 1) | 0 | Whether this node is a facility candidate. Each district must contain at least one candidate. |

### Optional

| Name | Type | Default | Purpose |
|---|---|---|---|
| `C_X` | float | 0.0 | X coordinate of the centroid (for visualization). |
| `C_Y` | float | 0.0 | Y coordinate of the centroid (for visualization). |
| `area` | float | 1.0 | Geographic area (used for compactness metrics). |
| `district` | int/str | — | Set by `Partition.write_to_graph()`. Used by `Partition.from_graph()` to reconstruct a partition. |
| `boundary_node` | bool | — | Set by `Graph.from_geodataframe()`. True if node is on the outer boundary. |
| `boundary_perim` | float | — | Length of the exterior boundary (for boundary nodes). |

## Edge attributes

These live on each edge: `graph.edges[u, v][attr]`.

| Name | Type | Default | Purpose |
|---|---|---|---|
| `shared_perim` | float | 1.0 | Shared boundary length between adjacent units. Used by rook adjacency and compactness. |

## Graph-level attributes

These live on the graph itself: `graph.graph[attr]`.

| Name | Type | Purpose |
|---|---|---|
| `crs` | str | Coordinate reference system (set by `from_geodataframe`). |
| `teams_per_district` | dict | District ID → team count. Set by `Partition.write_to_graph()`. |
| `capacity_level` | int | Max teams per district. Set by `Partition.write_to_graph()`. |

## Validation

All FalcomChain constructors validate the schema by default:

```python
graph = Graph.from_data(...)  # raises SchemaValidationError on missing attrs
graph = Graph.from_geodataframe(df)  # same
```

Disable validation with `validate=False` (not recommended for production).

Validate an existing graph manually:

```python
from falcomchain.graph import validate_graph, SchemaValidationError

try:
    validate_graph(graph, strict=True)
except SchemaValidationError as e:
    print(f"Graph schema problem: {e}")

# Or get a list of errors without raising
errors = validate_graph(graph, strict=False)
```

## Custom attributes

You can attach any extra attributes you want — the algorithms ignore them
unless you opt into using them (e.g., via a custom `energy_fn`).

```python
graph = Graph.from_data(
    edges=...,
    demand=...,
    candidates=...,
    extra_attributes={
        "vulnerability_index": {1: 0.8, 2: 0.3, ...},
        "service_type": {1: "clinic", 2: "hospital", ...},
    },
)
```
