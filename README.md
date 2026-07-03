# FalcomChain

[![tests](https://github.com/kirtisoglu/FalcomChain/actions/workflows/tests.yml/badge.svg)](https://github.com/kirtisoglu/FalcomChain/actions/workflows/tests.yml)
[![docs](https://github.com/kirtisoglu/FalcomChain/actions/workflows/docs.yml/badge.svg)](https://github.com/kirtisoglu/FalcomChain/actions/workflows/docs.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**FalcomChain** is a Python library for **hierarchical capacitated facility
location and districting problems** via Markov chain Monte Carlo. It samples
from the space of feasible plans, producing ensembles of contiguous districts
with facility assignments — useful for service zone design, sales territory
planning, healthcare network design, and stability analysis.

The library implements **FalCom** (Kaul & Kırtışoğlu), the first MCMC framework
that simultaneously partitions a region into capacity-respecting districts at
multiple hierarchy levels and assigns facilities, with convergence guarantees.

> **Status:** Pre-publication, under active development. The 0.1.0 API is
> stable but may evolve as the paper experiments solidify.

---

## What it does

Given a graph where nodes are geographic units with demand and facility
candidates, FalcomChain:

1. **Partitions** the graph into contiguous districts using capacitated
   spanning-tree cuts.
2. **Allocates** service teams to each district within a maximum capacity.
3. **Runs an MCMC chain** over the space of feasible hierarchical plans.
4. **Records** every chain step for ensemble analysis (boundary frequency,
   facility stability, capacity utilization).

---

## Installation

```bash
pip install falcomchain
```

For development:

```bash
git clone https://github.com/kirtisoglu/FalcomChain
cd FalcomChain
pip install -e ".[dev]"
```

Requires **Python 3.12+**.

---

## Documentation

Full documentation, tutorials, and API reference: **[falcomchain.readthedocs.io](https://falcomchain.readthedocs.io)** (coming soon).

In the meantime, browse the local docs:

|                                                       |                                                                  |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| [Getting started](docs/getting_started.md)            | 5-minute tutorial                                                |
| [Algorithm overview](docs/algorithm.md)               | What FalCom does, conceptually                                   |
| [Graph schema](docs/schema.md)                        | Required and optional graph attributes                           |
| [GeoDataFrame guide](docs/geodataframe.md)            | Building graphs from shapefiles/GeoJSON                          |
| [Candidate feasibility](docs/feasibility.md)          | Verify Assumption 6.1 and add artificial candidates              |
| [Ensemble analysis](docs/ensemble.md)                 | Boundary, facility, and capacity statistics across MCMC samples  |
| [Level-2 facilities](docs/super_facility.md)          | Opt-in super-facility assignment (Eq. 18) with pluggable selector |
| [Fixed superdistricts](docs/fixed_superdistricts.md)  | Hold the level-2 partition fixed (e.g. health zones)             |
| [Code structure](docs/structure.md)                   | Module-by-module breakdown                                       |
| [Tutorials](docs/tutorials/)                          | Jupyter notebook walkthroughs                                    |

---

## Companion libraries

FalcomChain is one of three small, decoupled libraries that together
cover the FalCom workflow. You can use FalcomChain alone, or compose
with the others as your workflow needs them.

| Library                                                            | Purpose                                                                                                                    | Status              |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| **FalcomChain** *(this library)*                                   | MCMC sampler for hierarchical capacitated facility location and districting.                                               | Active              |
| **[FalcomTravel](https://github.com/kirtisoglu/FalcomTravel)**     | Travel-time matrix computation (real travel times, graph distance, Euclidean) — feeds `Assignment.travel_times`.           | Planned             |
| **[FalcomPlot](https://github.com/kirtisoglu/FalcomPlot)**         | Static plotting for synthetic grids and interactive Leaflet maps for real geographies. Used by the FalcomChain doc pages.  | Active (pre-PyPI)   |

Typical end-to-end flow: build a `Graph` from your geodata, compute a
travel-time matrix with FalcomTravel, run a chain with FalcomChain,
visualize results with FalcomPlot.

---

## Architecture

![FalcomChain workflow](docs/falcomchain_workflow.drawio.png)

---

## Citation and acknowledgment

If you use FalcomChain in your research, please cite the paper:

```bibtex
@article{kaul2026falcom,
  title={FalCom: An MCMC Sampling Framework for Facility Location and Districting Problems},
  author={Kaul, Hemanshu and K{\i}rt{\i}{\c{s}}o{\u{g}}lu, Alaittin},
  year={2026},
}
```

FalcomChain's spanning-tree partition machinery and ReCom-style
proposal architecture are built on the foundation laid by
**[GerryChain](https://github.com/mggg/GerryChain)** (MGGG Redistricting Lab).
We extend GerryChain's redistricting framework to hierarchical,
capacitated problems with two-level facility assignment, and reuse the
log-proposal-ratio insight from
[Cannon, Duchin, Randall, Rule (2022)](https://arxiv.org/abs/2008.08054).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE.txt](LICENSE.txt).
