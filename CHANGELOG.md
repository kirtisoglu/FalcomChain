# Changelog

All notable changes to FalcomChain are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Balance rules named and checked.** The default rule keeps its behaviour
  and its name (`rule="per_team"`): the debt-clipped per-team window
  `[L~, U~]` with the *same absolute half-width* `(U~ - L~)/2` for every
  capacity `c`. The paper's capacity-scaled window `c · [L~, U~]` is available
  as `rule="scaled"` for comparison. The absolute rule moves the debt by at
  most one tolerance per extraction and keeps `|delta| <= tau` for every
  `c <= 3`, which is why a recursion with `c_max <= 3` (the London calibration)
  never meets an empty window; under the scaled rule the debt can move by
  `c · tau` and the recursion frequently fails at `c_max = 3` (on a 12x12
  test grid 12/12 seeds close under the absolute rule versus 7/12 under the
  scaled one; on the London instance initialization failed under the scaled
  rule). `tests/test_debt_rule.py` verifies containment, the debt-evolution
  interval, the `|delta| <= tau` invariant and telescoping on the instrumented
  recursion. The paper's Section 5.3 and Appendix A describe this absolute
  rule.
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
- **Residual-feasibility predicate at the supergraph**
  (`check_super_residual=True` in `CutParams`, `bipartition_tree`,
  `capacitated_recursive_tree`, `resample_super_partition` and
  `hierarchical_recom`). A one-sided level-2 extraction is admissible only if
  the supernodes it leaves behind can still be cut into super-districts with
  capacity in `[c2_min, c2_max]` holding at least `min_districts_super`
  districts each (`ceil(c/c_max) <= k <= min(floor(c/c_min), floor(n/kappa))`
  for some number `k` of super-districts). A necessary condition, so the
  feasible state space is unchanged; it removes the stranded-supernode
  rejections that dominated before: on the London real-station instance the
  acceptance rate rises from 38% to 68%, on the synthetic grids from about
  60% to over 90%, and steps get cheaper because the recursion no longer
  spends its retry budget on doomed residuals. `tests/test_super_residual.py`.
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

- Zone-based initialization (`Partition.from_random_assignment(super_assignment=...)`)
  cuts each zone against its own per-team target (`local_target=True`,
  paper Remark A.5), so the debt telescopes to zero inside every zone. On
  the London instance this initializes the 66-station problem in seconds
  where the global recursion often fails.
- Docs: the algorithm page no longer cites a convergence theorem the paper
  does not contain; level-2 facilities are described as the 1-median;
  Assumption 6.1 is presented as a sufficient condition and candidate repair
  as an optional initialization aid.
- Docs: the ensemble page's convergence section now follows the paper's
  validation program (exact enumeration on a 3x4 grid, start-independence
  of independently started chains, forgetting curves; Gelman--Rubin and ESS
  as secondary numbers) and shows the real-station London chains.

### Fixed
- The kappa constraint (each super-district holds at least
  `min_districts_super` districts) was enforced at the supergraph cut but not
  after the base-level re-cut, which could merge a super-district's districts
  into fewer than kappa; such proposals are now rejected
  (`SuperDistrictTooSmall`, cause `super_district_below_kappa`).
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
