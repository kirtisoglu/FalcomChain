# Working with GeoDataFrames

Most users will start from a shapefile, GeoPackage, or GeoJSON of geographic
units. This guide shows how to build a FalcomChain `Graph` from such data.

## Quick reference

```python
from falcomchain import Graph

# From a file
graph = Graph.from_file(
    "districts.shp",
    demand_col="POP",          # your population column
    candidate_col="is_clinic",  # your candidate flag column
)

# From a GeoDataFrame already loaded
import geopandas as gpd
gdf = gpd.read_file("districts.shp")
graph = Graph.from_geodataframe(
    gdf,
    demand_col="POP",
    candidate_col="is_clinic",
)
```

The library will:

1. Compute rook adjacency from polygon boundaries
2. Add `area` from each polygon's geometry
3. Add `shared_perim` to each edge
4. Add `boundary_node` and `boundary_perim` for outer-boundary units
5. Copy `demand_col` → `demand` and `candidate_col` → `candidate`
6. Validate the schema

## Specifying columns

If your demand column isn't named `demand`:

```python
graph = Graph.from_geodataframe(gdf, demand_col="population_2020")
```

The column is renamed to `demand` on the resulting graph (the canonical name
the algorithms use).

Same for candidates:

```python
graph = Graph.from_geodataframe(gdf, candidate_col="has_clinic_site")
```

## Marking facility candidates

The `candidate` column should be 0/1 or True/False per row. If you don't have
this column, create it before building the graph:

```python
# Example: any unit within 500m of a hospital is a candidate
gdf["candidate"] = (gdf.distance_to_hospital < 500).astype(int)
graph = Graph.from_geodataframe(gdf)
```

## Adding extra columns

Pass `cols_to_add` to copy additional columns as node attributes:

```python
graph = Graph.from_geodataframe(
    gdf,
    demand_col="POP",
    candidate_col="has_clinic",
    cols_to_add=["median_income", "vulnerability_score", "GEOID"],
)

# Now accessible as:
graph.nodes[0]["median_income"]
```

## CRS and reprojection

The library reads `gdf.crs` and stores it on `graph.graph["crs"]`. If your
data is in degrees (longitude/latitude), area and perimeter will be in
square degrees — usually wrong.

```python
# Reproject to a UTM projection automatically (good for area calculations)
graph = Graph.from_file("districts.shp", reproject=True)

# Or override the CRS if your file is missing it
graph = Graph.from_file("districts.shp", crs_override="EPSG:4326")
```

## Common errors

### `SchemaValidationError: Missing node attribute 'demand'`

You need a `demand` column in your GeoDataFrame, OR pass `demand_col="..."`
with the actual column name.

### `SchemaValidationError: No nodes are marked as facility candidates`

Your `candidate` column has all zeros (or is missing). At least one node
must be a candidate for the algorithms to produce valid districts.

### `GeometryError: Invalid geometries at rows ...`

Some polygons have self-intersections or other geometry issues. Either:
- Fix them: `gdf.geometry = gdf.geometry.buffer(0)`
- Or skip validation: `Graph.from_geodataframe(gdf, ignore_errors=True)`

### Areas/perimeters look weird

You're probably in lat/lon coordinates. Use `reproject=True`.

## Small example

```python
import geopandas as gpd
from shapely.geometry import box
from falcomchain import Graph

# 2x2 grid of unit squares
gdf = gpd.GeoDataFrame(
    {
        "POP": [100, 200, 150, 175],
        "is_clinic": [1, 0, 1, 0],
        "name": ["A", "B", "C", "D"],
    },
    geometry=[box(0,0,1,1), box(1,0,2,1), box(0,1,1,2), box(1,1,2,2)],
    crs="EPSG:32616",  # UTM zone — uses meters
)

graph = Graph.from_geodataframe(
    gdf,
    demand_col="POP",
    candidate_col="is_clinic",
    cols_to_add=["name"],
)

print(graph.nodes[0])
# {'area': 1.0, 'demand': 100, 'candidate': 1, 'name': 'A',
#  'boundary_node': True, 'boundary_perim': 4.0}
```
