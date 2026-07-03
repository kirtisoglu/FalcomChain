# FalcomChain Documentation

A Python library for hierarchical capacitated facility location and districting via MCMC.

## The FalCom family

FalcomChain is one of three companion libraries that together cover the
FalCom workflow. You can use them independently or compose them:

| Library                                                              | Purpose                                                                              | Status            |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| **FalcomChain** *(this library)*                                     | MCMC sampler for hierarchical capacitated facility location and districting.         | Active            |
| **[FalcomTravel](https://github.com/kirtisoglu/FalcomTravel)**       | Travel-time matrix computation, feeding `Assignment.travel_times`.                   | Planned           |
| **[FalcomPlot](https://github.com/kirtisoglu/FalcomPlot)**           | Static grid plots and interactive Leaflet maps; used by the executable doc pages.    | Active (pre-PyPI) |

FalcomChain's spanning-tree ReCom-style proposal architecture is built
on the foundation laid by
[GerryChain](https://github.com/mggg/GerryChain) (MGGG Redistricting
Lab). We extend GerryChain's redistricting framework to hierarchical,
capacitated problems with two-level facility assignment.

## User documentation

```{toctree}
:maxdepth: 2
:caption: Guides

getting_started
algorithm
schema
geodataframe
working_with_real_data
running_a_chain
optimization_methods
feasibility
ensemble
super_facility
fixed_superdistricts
```

## Tutorials

```{toctree}
:maxdepth: 1
:caption: Tutorials
:glob:

tutorials/*
```

## API reference

```{toctree}
:maxdepth: 2
:caption: API

api/falcomchain
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
