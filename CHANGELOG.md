# Changelog

All notable changes to FalcomChain are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **The paper's debt-corrected window is the default balance rule**
  (`rule="paper"`): a district at capacity `c` is admissible iff its demand
  lies in `c · [L~, U~]`, the debt-clipped per-team window scaled by `c`
  (paper eqs. (Li)-(Ui)). The previous default `"per_team"` used an
  absolute half-width for every `c` and was therefore `c` times stricter
  than the paper for `c >= 2`; it and `"main"` remain selectable as legacy
  rules. `tests/test_debt_rule.py` checks Theorem A.1, Corollaries A.2-A.3
  and Remark A.5 numerically on the instrumented recursion.
- **Cut selection is uniform over admissible cuts at `gamma = 0`** at both
  levels, as the paper describes. Previously level-1 cuts were weighted by
  the number of candidates in the subtree (and by the product of both
  sides' counts in two-sided mode) and level-2 cuts by their capacity;
  neither weighting was stated anywhere. `psi = 1[admissible] · exp(-gamma·eta)`;
  the complement's score uses the complement's own `eta` instead of a proxy.
- **Counting predicate is the default level-1 admissibility rule**
  (`count_candidates=True` in `CutParams`, `bipartition_tree`,
  `capacitated_recursive_tree`, `hierarchical_recom(count_candidates_base=True)`
  and `Partition.from_random_assignment`). The residual of every extraction
  must keep at least `ceil(remaining_teams / c_max)` candidates, and a
  one-sided cut must leave a candidate on both sides. This is what lets the
  chain run on sparse, real-world candidate sets (e.g. the 66 London
  ambulance stations) without artificial candidates; `repair_facility_density`
  remains available as an optional initialization aid.
- **Rejection accounting.** `MarkovChain` records why proposals were
  rejected (`chain.rejections`, `chain.last_rejection`,
  `chain.rejection_report()`, `classify_rejection`). Proposal-internal
  failures derive from `ProposalRejected` (a `RuntimeError`):
  `CutSearchExhausted(level, attempts)` when the retry budget runs out at the
  base or supergraph level, and `PopulationBalanceError`, which used to
  crash the chain.
- The level-2 facility selector is named for what it computes:
  `median_super_selector` (demand-weighted 1-median). `minimax_super_selector`
  is kept as an alias.

### Fixed
- In two-sided mode the root cut could extract the whole residual with fewer
  teams than remained, leaving an empty residual (surfaced as `IndexError`
  on the next draw). The root cut now has to absorb all remaining capacity,
  one-sided mode never extracts the root subtree, and an empty residual is a
  rejected proposal.
- `import falcomchain.tree.tree as ...` failed because the package
  star-imports leaked a `tree` name that shadowed the subpackage.
- `compute_energy` accepts states without a `super_facility` attribute.
- Zone-based initialization raises a clear `ValueError` when a zone's demand
  is below what one district of minimum capacity needs within tolerance.
- Test suite repaired (253 passing); CI now runs the whole suite. Two
  scaffold test files that never parsed were removed.

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
