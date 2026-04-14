# Repository Structure

```
falcomchain/          # Core library (the installable package)
experiments/          # Experiment runners and demos (not installed)
tests/                # Unit tests
docs/                 # User documentation
examples/             # Runnable examples
```

`data/` and `not_used/` are gitignored. Personal/research-only files live in
the user's separate `Brain/` repository.

---

## `falcomchain/` — Core library

### `random.py`
Dedicated `random.Random` instance and `set_seed()` for reproducible chains.
All stochastic operations in the library use this RNG instead of the global
`random` module.

### `graph/`

| File | Description |
|------|-------------|
| `graph.py` | `Graph` class (NetworkX subclass) with `from_data()`, `from_geodataframe()`, `from_file()`, `create()` smart constructor; `FrozenGraph` (immutable wrapper used during chain runs). |
| `schema.py` | Single source of truth for required/optional node, edge, and graph attributes. `validate_graph()`, `describe_schema()`, `SchemaValidationError`. |
| `adjacency.py` | Rook/queen adjacency between geographic units. |
| `geo.py` | CRS reprojection utilities, geometry validation. |
| `grid.py` | `Grid` synthetic graph (testing/demo utility). |
| `metagraph.py` | Supergraph construction utilities. |

### `partition/`

| File | Description |
|------|-------------|
| `partition.py` | `Partition` (state object), `Partition.from_random_assignment()`, `from_graph()`, `write_to_graph()`, `save_json()`, `load_json()`. |
| `assignment.py` | `Assignment` — node→district mapping, district candidates, teams, travel-time class attribute. |
| `flows.py` | `Flow` — diffs between successive partitions (part_flows, node_flows, candidate_flows). |
| `cut_edges.py` | Edges crossing district boundaries. |
| `subgraphs.py` | `SubgraphView` — lazy view of district subgraphs. |
| `compactness.py` | Partition-level compactness metrics. |

### `tree/`

| File | Description |
|------|-------------|
| `tree.py` | `SpanningTree`, `CutParams`, `Cut`, `Flip`. `uniform_spanning_tree()` (Wilson's algorithm — default), `random_spanning_tree()` (Kruskal-random). `bipartition_tree()`, `capacitated_recursive_tree()`. The `tree_sampler`, `psi_fn`, and `recorder` parameters are pluggable. |
| `errors.py` | `ReselectException`, `PopulationBalanceError`, `BalanceError`, `BipartitionWarning`. |
| `snapshot.py` | `Recorder` class. Binary delta-encoded format (`chain.fcrec`) + per-step phase JSON for animation. `Recorder.export_to_json()`. |

### `markovchain/`

| File | Description |
|------|-------------|
| `chain.py` | `MarkovChain` iterator. Optional `recorder` parameter for animation output. |
| `proposals.py` | `hierarchical_recom()` (paper's Algorithm 1: recursive partitioning at supergraph + base levels). Also `recom()`, `propose_random_flip()`, `propose_chunk_flip()`. |
| `state.py` | `ChainState` — wraps `Partition` with energy, log_proposal_ratio, beta, feasibility, FacilityAssignment, SuperFacilityAssignment, optional energy_fn. |
| `accept.py` | `always_accept()` (paper default), `metropolis_hastings()` (Boltzmann optimizer mode). |
| `energy.py` | `compute_energy()` — demand-weighted travel time. `compute_energy_delta()`. |
| `facility.py` | `FacilityAssignment` (level-1 minimax centers), `SuperFacilityAssignment` (level-2). |
| `objectives.py` | `squared_radius_deviation()`, `total_cut_edges()`, `polsby_popper()`. |
| `optimization.py` | `SingleMetricOptimizer`. |

### `constraints/`

| File | Description |
|------|-------------|
| `validity.py` | `Validator`, demand-balance checks. |
| `contiguity.py` | Contiguity constraint functions. |
| `compactness.py` | Compactness constraints. |
| `bounds.py` | `LowerBound`, `UpperBound`, `Bounds`. |

### `candidates/`

| File | Description |
|------|-------------|
| `candidates.py` | `Block`, `Cell`, `Cells` — spatial grid for sampling facility candidates. |

### `tally/`

| File | Description |
|------|-------------|
| `tally.py` | `Tally` — accumulates per-step statistics. |
| `logger.py` | Logging utilities. |

### `helper/`

| File | Description |
|------|-------------|
| `data_handler.py` | `DataHandler`, `save_pickle()`, `load_pickle()`. |

### `vendor/`
Bundled third-party code (UTM coordinate conversion).

---

## Pluggable components

The library is designed for researchers to test variants from the paper's
future-work section. These are the main extension points:

| Component | How to customize | Default | Variants enabled |
|-----------|-----------------|---------|-----------------|
| Spanning tree sampling | `bipartition_tree(tree_sampler=...)` | `uniform_spanning_tree` (Wilson's) | Weighted spanning trees |
| Candidate-awareness score | `CutParams(psi_fn=...)` | `phi * exp(-gamma * r)` | Adaptive gamma, custom scoring |
| Energy function | `ChainState.initial(energy_fn=...)` | `compute_energy` | Equity objectives, domain-specific |
| Acceptance rule | `MarkovChain(accept=...)` | `always_accept` | MH (Boltzmann optimizer), custom |
| Distance metric | `Assignment.travel_times` | None | Real travel times, graph distance, Euclidean |

---

## Schema

See [schema.md](schema.md) for the full attribute schema. Key points:

**Required node attributes:**

| Name | Type | Purpose |
|------|------|---------|
| `demand` | float | Demand per unit |
| `candidate` | int (0/1) | Facility candidate flag |

**Set by `Partition.write_to_graph()`** (so the graph alone fully describes a partition):

| Name | Scope | Purpose |
|------|-------|---------|
| `district` | node | Current district assignment |
| `teams_per_district` | graph-level | Dict district → team count |
| `capacity_level` | graph-level | Max teams per district |

**Supergraph node attributes** (set by `supergraph()` builder):

| Name | Description |
|------|-------------|
| `demand` | Aggregate demand of all base units in the district |
| `area` | Aggregate area |
| `n_teams` | Service teams assigned |
| `n_candidates` | Number of candidate facility sites |
