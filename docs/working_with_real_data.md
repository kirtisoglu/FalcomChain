# Case Study: Working with Real Data

```{admonition} Goal of this page
:class: tip
The end-to-end recipe for a real instance — the London Ambulance
Service — from raw geodata and travel times through feasibility repair
to a recorded chain, with links to the focused reference for each step.
```

The synthetic demo grid is fine for tutorials, but real workflows
start from shapefiles, GeoPackages, or GeoJSON of geographic units
and use real travel times. This page is a workflow guide that ties
together the pieces you need, with links to the focused references
for each.

The running example throughout is the **London Ambulance Service
(LAS)** application: 5,042 LSOAs (Lower Layer Super Output Areas)
covering Greater London, ~1.06 M annual incidents as demand,
63 real ambulance stations as level-1 candidates, and 7 HQ/EOC sites
as level-2 candidates. The full LAS notebook lives in the
[London-Ambulance-Service-System](https://github.com/kirtisoglu/London-Ambulance-Service-System)
repo; this page distills the recipe.

## 0. Visual overview

Before any chain mechanics, it helps to see what the dataset actually
looks like. The figure below is built with FalcomPlot's mapping
helpers — `build_basemap`, `add_hierarchy`, and `add_markers` — and
shows the three layers users will reach for repeatedly: the LSOA
polygons (`G¹` base), dissolved borough boundaries (administrative
reporting layer, *not* a chain level), and the L1/L2 facility
candidates.

![Greater London — LSOAs, boroughs, and LAS facilities](_static/london_overview.png)

Recipe (paths assume you cloned the LAS repo as a sibling directory):

```python
import geopandas as gpd
import pandas as pd
from falcomplot.mapping import plott

lsoa = gpd.read_file("data/raw/LSOA_2021_London.gpkg").to_crs(4326)
lsoa["borough"] = lsoa["LSOA21NM"].str.replace(r" \d{3}[A-Z]$", "", regex=True)

stations = pd.read_csv("data/raw/LAS_stations.csv")
stations_gdf = gpd.GeoDataFrame(
    stations.assign(category="LAS station", source="LAS",
                    name=stations["station_name"]),
    geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
    crs=4326,
)

l2 = pd.read_csv("data/raw/LAS_L2_facilities.csv")
l2["category"] = l2["facility_type"].map({
    "EOC": "Emergency Operations Centre",
    "Sector HQ": "Sector HQ (proxy)",
}).fillna("Sector HQ (proxy)")
l2["source"] = "LAS"
l2["name"] = l2["facility_name"]
l2_gdf = gpd.GeoDataFrame(
    l2,
    geometry=gpd.points_from_xy(l2["longitude"], l2["latitude"]),
    crs=4326,
)

c = lsoa.geometry.union_all().centroid
m = plott.build_basemap(boundary=lsoa, center=(c.y, c.x), zoom=10)
plott.add_hierarchy(
    m, hierarchy="borough", basemap_gdf=lsoa,
    color="#222", weight=1.6, fill_color="#fed8b1", fill_opacity=0.18,
    tooltip_fields=["borough"],
)
categories = {
    "LAS station": {"color": "#e63946", "radius": 5, "order": 0},
    "Emergency Operations Centre": {"color": "#1d3557", "radius": 8, "order": 1},
    "Sector HQ (proxy)": {"color": "#457b9d", "radius": 7, "order": 2},
}
plott.add_markers(m, stations_gdf, categories=categories)
plott.add_markers(m, l2_gdf, categories=categories)
m.save("london_overview.html")
```

The interactive Leaflet output lets you hover for borough labels and
click markers for facility metadata. For the static figure above the
LAS repo ships
[`analysis/run_overview_map.py`](https://github.com/kirtisoglu/London-Ambulance-Service-System/blob/main/analysis/run_overview_map.py),
which produces both the HTML and the PNG.

## The end-to-end pipeline

```
Geographic data            Travel times
(shapefile/GeoJSON)        (FalcomTravel)
        │                         │
        ▼                         ▼
   Graph.from_file()      Assignment.travel_times = ...
        │                         │
        └────────────┬────────────┘
                     ▼
        check_facility_density()         ← Assumption 6.1
                     │
              [repair if needed]
                     ▼
       Partition.from_random_assignment()
                     │
                     ▼
              ChainState.initial()
                     │
                     ▼
              MarkovChain(...)
                     │
              [callbacks: ensemble, recorder]
                     ▼
                 .report() / animation
```

## 1. Build the graph from your data

Use `Graph.from_file` for shapefiles / GeoPackages / GeoJSON, or
`Graph.from_geodataframe` if you already have a `GeoDataFrame` in
memory. Both accept the same column kwargs.

```python
from falcomchain import Graph

graph = Graph.from_file(
    "districts.shp",
    demand_col="POP",                # column holding demand (population, workload, etc.)
    candidate_col="is_clinic",        # 0/1 column for level-1 candidates
    super_candidate_col="is_hub",     # 0/1 column for level-2 candidates (optional)
    reproject=True,                   # convert lat/lon to UTM for area/perimeter
)
```

The full reference, including extra columns, CRS handling, and common
errors, lives in [Working with GeoDataFrames](geodataframe.md). The
attribute schema (what `demand`, `candidate`, `super_candidate` mean)
is in [Graph Attribute Schema](schema.md).

## 2. Compute travel times with FalcomTravel

Travel times feed `Assignment.travel_times` and drive the level-1
covering radius (Eq. 17) and the optional energy objective
(demand-weighted travel time). For real road networks,
[FalcomTravel](https://github.com/kirtisoglu/FalcomTravel) produces the
matrix from OSM data (multimodal via r5r, or pure-Python via OSMnx):

```python
import falcomtravel as ft
from falcomchain.partition.assignment import Assignment

coords = ft.coords_from_graph(graph, x="C_X", y="C_Y")   # (lon, lat)
backend = ft.R5RBackend(
    data_path="data/london",          # OSM .pbf + GTFS .zip
    coords=coords,
    timezone="Europe/London",
)
matrix = backend.compute(
    origins=station_nodes,
    destinations=list(coords),
    mode=ft.Mode.DRIVE,
)
matrix.to_parquet("travel_times.parquet")     # compute once, reuse

Assignment.travel_times = matrix.as_dict()
```

The full backend guide — including `diagnose()` for catching
disconnected road fragments before an hour-long routing run, and the
dependency-free fallbacks for prototyping — is on the
[Travel Times with FalcomTravel](travel_times.md) page.

## 3. Verify Assumption 6.1

Before running a chain on real data, verify that the candidate set is
dense enough that no facility-free region can form a valid district.
A real candidate set with sparse coverage will trap the chain.

```python
from falcomchain import check_facility_density, repair_facility_density

report = check_facility_density(graph, demand_target=50_000, epsilon=0.10, c_min=2)
if not report.passes:
    print(f"Adding artificial candidates to repair {len(report.violating_components)} components")
    added = repair_facility_density(
        graph,
        demand_target=50_000,
        epsilon=0.10,
        c_min=2,
        strategy="weighted_center",
    )
    # Each added node now has graph.nodes[n]["candidate_artificial"] = 1
    # so downstream tools (e.g. ensemble analysis) can distinguish synthetic
    # candidates from real ones.
```

**LAS reality check.** On the London graph, the 63 real stations do
*not* fragment the LSOA graph — removing them leaves one connected
component of 4,979 nodes carrying 9.05 M demand. Assumption 6.1 fails
at every operational `(demand_target, c_min)` combination, so the
repair step is mandatory. See
[`analysis/01_feasibility.md`](https://github.com/kirtisoglu/London-Ambulance-Service-System/blob/main/analysis/01_feasibility.md)
for the full report.

Full details and strategy comparison: [Candidate Feasibility](feasibility.md).

## 4. Build the initial partition

```python
from falcomchain import Partition

partition = Partition.from_random_assignment(
    graph=graph,
    epsilon=0.1,
    demand_target=1500,
    assignment_class=None,
    capacity_level=2,
    # super_assignment={node: zone_id}    # if zones are externally given
    # init_super_partition=True,           # paper-faithful initial level-2 partition
    # epsilon_super=0.15,                  # required when init_super_partition=True
)
```

By default the initial level-2 partition is identity (each level-1
district is its own superdistrict) — cheap, and fine because the
first `hierarchical_recom` step resamples it anyway. Pass
``init_super_partition=True`` to instead build the initial level-2
partition by recursively partitioning the supergraph (paper Section 5.1
"Initialization"). Useful when you want an honest initial state for
ensemble analysis from step 0.

## 5. Run the chain

The mechanics are the same as on the demo grid. See
[Running a FalCom Chain](running_a_chain.md) for the full walk-through;
the only difference for real data is scale (chains typically run
5,000–50,000 steps) and the optional recording layer for animations.

## 6. Save and reload chain state

A graph with the partition baked in is a complete description of a
chain state — you can serialize it as a single artifact:

```python
import pickle

# Save
partition.write_to_graph()  # writes 'district' attribute on each node
                            # plus 'teams_per_district' and 'capacity_level'
                            # on the graph itself
with open("state.pkl", "wb") as f:
    pickle.dump(graph, f)

# Reload (no need to recompute the partition)
with open("state.pkl", "rb") as f:
    graph = pickle.load(f)
partition = Partition.from_graph(graph)
```

For the partition alone (not the graph), use
`partition.save_json("partition.json")` and
`Partition.load_json("partition.json", graph)`.

## 7. Record for animation

To produce data that the
[FalcomPlot](https://github.com/kirtisoglu/FalcomPlot) JS animation
app can replay step by step:

```python
from falcomchain.tree.snapshot import Recorder

recorder = Recorder("output/", record_substeps=True)
recorder.write_header(graph, partition, params={
    "epsilon": 0.1,
    "demand_target": 1500,
    "capacity_level": 3,
})

chain = MarkovChain(..., recorder=recorder)
for state in chain:
    pass
recorder.close()

# Export to JSON for the dashboard
Recorder.export_to_json("output/", "output/json/")
```

The recorder writes a compact binary trajectory (`chain.fcrec`) plus
optional per-step substep JSON for the four-phase animation.

## 8. Real-data caveats

- **Missing `crs`** — if your shapefile has no CRS, areas and
  perimeters will be in degrees-squared (almost certainly wrong).
  Pass `reproject=True` to `from_file` or set `crs_override`.
- **Disconnected graph** — if the graph has multiple connected
  components (e.g., islands), partition each component separately or
  use `super_assignment` to keep them in distinct superdistricts.
- **Heavy-demand outlier nodes** — if a single node's demand exceeds
  `c_max · demand_target · (1 + ε)`, the algorithm cannot place it
  inside any admissible district. Either subdivide the unit, raise
  `epsilon`, or merge it with neighbors before building the graph.
- **Travel-time matrix mismatches** — `Assignment.travel_times` keys
  must match `graph.nodes` exactly. If FalcomTravel uses GEOIDs and
  your graph is built with sequential integers, build a key map
  before assignment.

## Where to go next

- [Running a FalCom Chain](running_a_chain.md) — full chain mechanics.
- [Optimization Methods](optimization_methods.md) — sampling vs
  Boltzmann optimization vs custom acceptance.
- [Ensemble Analysis](ensemble.md) — boundary, facility, and capacity
  statistics across samples, including the London Ambulance
  convergence diagnostics.
- [Visualization with FalcomPlot](visualization.md) — every plotting
  helper used above.
