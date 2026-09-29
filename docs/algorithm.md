# Algorithm Overview

```{admonition} Goal of this page
:class: tip
Understand what the chain does at each step and why sampling beats a
single optimum — no code, no proofs. For the full mathematical
treatment, see the FalCom paper.
```

## The problem

You have a geographic region divided into small basic units (census blocks,
postcodes, ZIP codes). You want to:

1. Group those units into **districts** that are contiguous and roughly
   demand-balanced
2. Assign **service teams** to each district up to a capacity limit
3. Place a **facility** in each district at one of a fixed set of candidate
   sites
4. Optionally do all of the above at multiple **hierarchy levels**
   (e.g., clinic → community hospital → regional medical center)

This is the **hierarchical capacitated facility location problem (HCFLP)**.
Classical optimization methods scale to ~1,000 units. FalCom samples plans
at 50,000+ units.

## Why MCMC?

A single "optimal" solution is brittle: small changes in demand or constraints
can flip facility placements. FalCom samples from the space of feasible
plans, producing an **ensemble**. From the ensemble you can ask:

- Which district boundaries are robust (appear in 90%+ of plans)?
- Which facilities are essential vs. substitutable?
- How does capacity utilization vary across plans?

This is the same paradigm political scientists use for redistricting fairness,
applied to facility location.

## The algorithm in one paragraph

FalCom builds on **ReCom** (Recombination, DeFord, Duchin, Solomon 2021).
At each step: pick two adjacent districts, merge them into one region,
sample a uniform random spanning tree of the merged region, and cut a
balanced edge to produce two new districts. Contiguity is guaranteed by
construction (cutting a tree edge always yields two connected components).
FalCom extends ReCom in three ways:

1. **Hierarchical proposal** — operate on a supergraph of districts, then
   re-split one selected superdistrict at the base level
2. **Capacitated tree cuts** — produce a variable number of districts under
   capacity constraints in a single recursive partitioning
3. **Candidate-aware cut selection** — bias toward cuts where a facility
   candidate is centrally located

## The four phases of one iteration

Each FalCom iteration has four named phases, called recursively at two levels:

```
Phase 1: Hierarchical Proposal
  └── Apply Phases 2 + 3 + 4 at the supergraph level
      └── Phase 2: Recursive Partitioning of G²
          └── Phase 3: Capacitated Tree Cut (repeated)
      └── Phase 4: Level-2 Facility Assignment
  └── Choose superdistrict D² uniformly at random
  └── Apply Phases 2 + 3 + 4 at the base level on G¹[D²]
      └── Phase 2: Recursive Partitioning
          └── Phase 3: Capacitated Tree Cut (repeated)
      └── Phase 4: Level-1 Facility Assignment
  └── Accept or reject the new state
```

### Phase 2: Recursive Partitioning

Extracts districts one at a time from a residual graph. Tracks **demand debt**
(cumulative deviation from target) and tightens the per-district demand bounds
so the final district falls within tolerance.

### Phase 3: Capacitated Tree Cut

For a residual graph H:
1. Sample a uniform spanning tree T of H (Wilson's algorithm)
2. For every node u in T, decide whether the subtree rooted at u is
   **admissible**: its demand fits the debt-corrected window of some
   capacity, it contains a facility candidate, and the residual it leaves
   keeps enough candidates for the districts still to be cut (the
   *counting predicate*, at least `ceil(remaining teams / c_max)`)
3. Score every admissible subtree with ψ(u) = exp(-γ · η(u)), where η(u)
   is the per-capita access cost of the best-located candidate
   (demand-weighted 1-median); at γ = 0 every admissible subtree scores 1
4. Select a subtree with probability proportional to ψ — uniformly over
   admissible cuts at γ = 0
5. The selected subtree becomes the next extracted district

### Phase 4: Facility Assignment

Deterministic: for each district, pick the candidate that minimizes the
demand-weighted total travel time to the district's units (the
demand-weighted 1-median). The same rule is applied one level up for the
level-2 facility of each superdistrict.

At the supergraph level the analogous check is the *residual-feasibility
predicate*: a one-sided level-2 extraction is admissible only if the
supernodes it leaves behind can still form super-districts with capacity in
`[c2_min, c2_max]` and at least `min_districts_super` districts each. Both
predicates are necessary conditions, so neither changes the feasible state
space; they keep the recursion from spending its retry budget on residuals
that cannot close.

## Convergence

No mixing-time or stationary-distribution result is known for FalCom, as
for the recombination chains it extends. The chain is validated
empirically, the way the redistricting literature does: exact enumeration
of all feasible states on small instances (support and start-independence
of the empirical distribution), and comparison of summary-statistic
distributions across chains started from very different plans. The
sampled distribution is shaped by the uniform spanning-tree distribution
on each re-cut region and by the candidate-awareness score ψ (controlled
by γ); at γ = 0 cut selection is uniform over admissible cuts.

## Acceptance

The default acceptance is `always_accept`. The chain is a sampler, not an
optimizer — it explores the feasible space rather than minimizing energy.

For optimization variants (find a low-energy plan), use the
`boltzmann` acceptance rule with a custom energy function. ``boltzmann``
is a heuristic optimizer, not a true Metropolis-Hastings sampler — it
omits the proposal-density ratio because that ratio is intractable for
FalCom (the standard ReCom MH formulation requires the
[RevReCom correction of Cannon et al. (SIAM Review, 2026)](https://arxiv.org/abs/2008.08054),
which we have not implemented).

## Cost of a step

A step draws spanning trees of the supergraph until the level-2 recursion
closes, then draws spanning trees of one merged superdistrict until the
level-1 recursion closes. Profiling 300-step chains on the paper's sparse
grids (seed 7, cProfile) puts the level-2 resampling at 1% of the step on
grid_1000 (about 12 districts), 24% on grid_10000 (about 40) and 31% on
grid_50000 (about 90): the supergraph term grows with the number of
districts but stays below a third of the step at ninety. Three things
decide the wall-clock time in practice:

- **Retries.** A proposal that cannot close spends its whole retry budget
  (`max_attempts`, 1,000 in the paper) drawing trees, so a rejected
  proposal costs up to a thousand times an accepted one. On instances with
  many rejections a smaller budget (100 to 200) cuts the run time
  substantially; it also changes the transition kernel slightly (proposals
  that would have closed late are rejected instead), so keep it fixed
  within one study and report it.
- **Tree draws.** Wilson's algorithm walks precomputed neighbour lists
  (since version 0.x the residual graph is converted once per bipartition
  call); a 300-step London chain went from 13.5 to 22.5 steps per second
  with this and two smaller changes, bit-identically.
- **Districts per call.** The recursion extracts districts one at a time
  from a shrinking residual, so a call that must produce hundreds of
  districts is slow and fails more often. Keep each call in the tens of
  districts: on London the initial partition is built sector by sector
  (11 to 17 stations each) and on the 50,000-node grid zone by zone
  (blocks of about 5,000 nodes); the chain's own steps re-cut one
  superdistrict of a few districts.

## Initial state

The initial partition is constructed by applying Phase 2 to the base graph,
either globally or zone by zone (`super_assignment=`), each zone against its
own per-team target. On large instances with few candidates the zone-by-zone
form is both faster and far more likely to close; the first accepted step
replaces the zone grouping by a sampled level-2 partition. The chain still
needs a burn-in before its samples are used.

## Where to next

- [Getting started](getting_started.md) — build and run your first chain
- [Running a FalCom Chain](running_a_chain.md) — the four phases above,
  as executable code
- [Ensemble Analysis](ensemble.md) — what the sampled ensemble buys you
- The **FalCom paper** for proofs, notation, and experiments
