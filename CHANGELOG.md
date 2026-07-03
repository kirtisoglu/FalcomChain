# Changelog

All notable changes to FalcomChain are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — TBD

Initial public release. See the FalCom paper (Kaul & Kırtışoğlu, 2026) for
the algorithm and theory.

### Added
- **Ensemble diagnostics in FalcomPlot.** A new `falcomplot.ensemble`
  module provides `plot_trace`, `plot_convergence`, and
  `plot_boundary_frequency`, plus the statistics `gelman_rubin`
  (split $\hat R$), `effective_sample_size`, and `cut_frequencies`.
  These are the convergence and boundary-frequency tools used in the
  FalCom paper; the Ensemble Analysis docs page now demonstrates them
  via `import falcomplot as fp`.
- **London Ambulance Service case study.** The paper's LAS instance
  (4,994-node LSOA dual graph, 66 stations, 5 sectors) is run at the
  capacity-block calibration; four independently seeded chains reach
  $\hat R \approx 1.00$, and the operational layout lands within 4.6% of
  the best matched-count configuration found.

### Notes
- The recursive partitioning phase is numerically stable in the safe
  capacity range $c \in \{1,2\}$; capacities $c \geq 3$ require the
  capacity-block reparametrization described in the paper (a per-team
  block bundles several units), as used for the LAS instance.
