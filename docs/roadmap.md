# FalCom Development Roadmap

## Completed

- Flow dataclass grouping node_flows, part_flows, candidate_flows
- SpanningTree refactored: CutParams separates tree structure from cut parameters
- MarkovChain iterates over ChainState (not Partition)
- Energy function: compute_energy (demand-weighted travel time) in energy.py
- FacilityAssignment: level-1 minimax centers computed on ChainState
- SuperFacilityAssignment: level-2 centers
- log_proposal_ratio computed in hierarchical_recom from psi scores
- Wilson's algorithm for uniform spanning tree sampling
- Real candidate-awareness score psi with demand radius
- Pluggable tree sampler, psi function, energy function
- Recorder class for animation output (replaces old snapshot exports)

## In Progress

- Experiments for paper (convergence diagnostics, scalability, case studies)
- FalcomPlot integration with new Recorder output format

## Planned

- Weighted spanning trees (edge weighting by distance, demand similarity, etc.)
- Adaptive gamma (simulated annealing schedule for candidate-awareness)
- |L| > 2 hierarchy levels
- Parallel chain support (multiprocessing + parallel tempering)
- Dashboard: load data, configure chain, run, animate
