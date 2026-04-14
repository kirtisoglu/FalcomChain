# Algorithm Overview

This page gives a conceptual overview of FalCom — what it does and how it
works at a high level. For the full mathematical treatment (theorems, proofs,
notation), see the **paper**.

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
2. For every node u in T, compute the **feasibility score** φ(u): is the
   subtree rooted at u demand-balanced AND containing a facility candidate?
3. Compute the **candidate-awareness score** ψ(u) = φ(u) · exp(-γ · r(u))
   where r(u) is the demand radius (eccentricity of the best candidate)
4. Select a cut edge with probability proportional to ψ
5. The selected subtree becomes the next extracted district

### Phase 4: Facility Assignment

Deterministic: for each district, pick the candidate that minimizes the
maximum travel time to any node in the district (minimax / covering radius).

## Convergence

FalCom is irreducible and aperiodic over the feasible state space, so it
converges to a unique stationary distribution π* (paper Theorem 6.3 and
Corollary 6.5). The distribution is shaped by:

- The **uniform spanning tree distribution** on each merged region
- The **candidate-awareness score** ψ (controlled by γ)

When γ = 0, ψ reduces to the feasibility count φ, and cut selection becomes
uniform over admissible cuts.

## Acceptance

The default acceptance is `always_accept`. The chain is a sampler, not an
optimizer — it explores the feasible space rather than minimizing energy.

For optimization variants (find a low-energy plan), use
`metropolis_hastings` with a custom energy function. Note: the standard
ReCom MH formulation does not satisfy detailed balance without a Cannon et al.
2022 reversibility correction; the current `metropolis_hastings` is a
soft-Boltzmann approximation suitable for optimization but not strict sampling.

## Initial state

The initial partition is constructed by applying Phase 2 directly to the
full base graph. This produces a feasible state in O(|V|) time, so the chain
can begin sampling immediately without burn-in.

## Where to next

- [Getting started](getting_started.md) — build and run your first chain
- [Schema reference](schema.md) — what attributes the algorithms read
- [GeoDataFrame guide](geodataframe.md) — work with shapefiles
- The **FalCom paper** for proofs, notation, and experiments
