# FalcomChain Documentation

FalcomChain is a Python library for **hierarchical capacitated facility
location and districting via MCMC**: it partitions a geographic region
into contiguous, demand-balanced districts, places facilities from a
candidate set, and — instead of returning one brittle "optimum" —
samples an *ensemble* of near-optimal plans you can interrogate for
robust boundaries, essential facilities, and capacity utilization.

FalcomChain's spanning-tree ReCom-style proposal architecture builds on
[GerryChain](https://github.com/mggg/GerryChain) (MGGG Redistricting
Lab), extended to hierarchical, capacitated problems with two-level
facility assignment.

## The FalCom family

Three companion libraries cover the full workflow. This documentation
covers all three: FalcomTravel and FalcomPlot each have a dedicated page
here rather than separate doc sites.

| Library | Role in the pipeline | Documented in |
| --- | --- | --- |
| **[FalcomTravel](https://github.com/kirtisoglu/FalcomTravel)** | Geodata → travel-time matrix (r5r, OSMnx, Dijkstra, Euclidean backends) | [Travel Times with FalcomTravel](travel_times.md) |
| **FalcomChain** *(this library)* | Travel times + graph → MCMC ensemble of districting plans | everything else |
| **[FalcomPlot](https://github.com/kirtisoglu/FalcomPlot)** | Ensemble → plots, diagnostics, animations, interactive maps | [Visualization with FalcomPlot](visualization.md) |

## How the documentation is organized

Each page has one specific goal, stated at the top. The groups below
follow the order of a real project: understand the method, prepare your
inputs, run chains, analyze the output.

```{toctree}
:maxdepth: 1
:caption: Start here

getting_started
algorithm
```

```{toctree}
:maxdepth: 1
:caption: Prepare your inputs

geodataframe
travel_times
feasibility
```

```{toctree}
:maxdepth: 1
:caption: Run the chain

running_a_chain
super_facility
optimization_methods
fixed_superdistricts
```

```{toctree}
:maxdepth: 1
:caption: Analyze the ensemble

ensemble
visualization
```

```{toctree}
:maxdepth: 1
:caption: Case study

working_with_real_data
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
:glob:

tutorials/*
```

```{toctree}
:maxdepth: 1
:caption: Project

reproducibility
contributing
```

```{toctree}
:maxdepth: 1
:caption: Reference

schema
api/falcomchain
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
