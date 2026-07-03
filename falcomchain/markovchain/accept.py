import math
from typing import Callable

from falcomchain.markovchain.state import ChainState
from falcomchain.random import rng


def always_accept(_proposed: ChainState, _current: ChainState) -> bool:
    """
    Acceptance function that accepts every proposed state unconditionally.

    This is the paper's intended acceptance rule (Section 6.3): a proposal
    is feasible iff every cut has ψ > 0, and feasible proposals are
    accepted unconditionally. Use this for ensemble sampling.

    :param proposed: The proposed chain state.
    :param current: The current chain state (unused).
    :returns: Always ``True``.
    :rtype: bool
    """
    return True


def boltzmann(state: ChainState, parent: ChainState) -> bool:
    """
    Boltzmann acceptance rule for energy-biased optimization.

    Accepts the proposed state with probability

        alpha(s -> s') = min(1, R(s') * exp(-beta * (E(s') - E(s))))

    where:

    - ``R(s')`` is the hard feasibility indicator → ``state.feasible``
    - ``E(s')`` is the energy of the proposed state → ``state.energy``
    - ``E(s)`` is the energy of the current state → ``parent.energy``
    - ``beta`` is the inverse temperature → ``state.beta``

    Requires the user to have explicitly opted into an objective by
    passing ``energy_fn=...`` to :meth:`ChainState.initial`. Without an
    energy function FalCom is a sampler — call this only when you've
    decided to optimize.

    .. note::

        This is **not** a true Metropolis-Hastings sampler. It omits the
        proposal-density ratio ``q(s|s')/q(s'|s)`` because that ratio is
        intractable for FalCom (paper Section 6.3): it depends on the
        number of admissible cuts across all spanning trees of the merged
        subgraph in both the current and proposed states, which is #P-hard
        to compute (Jerrum & Sinclair). As a result this function is a
        **heuristic optimizer** that drives the chain toward low-energy
        states, not a sampler from a Boltzmann distribution. For sampling
        with convergence guarantees, use :func:`always_accept`.

    :param state:  The proposed ChainState.
    :param parent: The current ChainState.
    :returns: True if the proposal is accepted, False otherwise.
    :rtype: bool
    :raises RuntimeError: If ``state.energy_fn`` is ``None`` — this would
        mean Boltzmann is being called without an objective, which would
        silently degrade to ``always_accept``. Set ``energy_fn`` on
        ``ChainState.initial`` (e.g.
        ``energy_fn=falcomchain.markovchain.energy.compute_energy``).
    """
    if state.energy_fn is None:
        raise RuntimeError(
            "boltzmann acceptance requires state.energy_fn to be set. "
            "FalCom defaults to a sampler with no objective. To use the "
            "Boltzmann optimizer, pass an explicit energy function:\n\n"
            "    from falcomchain.markovchain.energy import compute_energy\n"
            "    state = ChainState.initial(..., energy_fn=compute_energy)\n\n"
            "For pure sampling without an optimizer, use always_accept "
            "instead of boltzmann."
        )

    if not state.feasible:
        return False

    if state.beta == 0:
        return True

    delta_energy = state.energy - parent.energy
    log_alpha = -state.beta * delta_energy

    return math.log(rng.random()) <= log_alpha


def soft_constraint_accept(
    violation_fn: Callable[[ChainState], float],
    beta_feas: float = 1.0,
    base_accept: Callable[[ChainState, ChainState], bool] = always_accept,
) -> Callable[[ChainState, ChainState], bool]:
    """
    Build an acceptance rule that softly penalises *per-state* constraint
    violations.

    Returns an ``accept(proposed, current)`` callable that:

    1. Calls ``base_accept(proposed, current)`` first.
    2. If accepted, computes ``v = violation_fn(proposed)`` (≥ 0).
    3. Multiplies the acceptance probability by ``exp(-beta_feas · v)``.

    The chain's stationary distribution becomes (informally) proportional
    to ``π_base(s) · exp(-beta_feas · v(s))``, so violating states have
    vanishing mass as ``beta_feas → ∞`` and equal mass as
    ``beta_feas → 0``.

    .. warning::

        ``violation_fn`` **must** be state-dependent for this to bias
        the chain. If it returns the same value regardless of state
        (e.g. a function of fixed graph attributes), every proposal
        is multiplied by the same constant ``exp(-beta_feas · v)`` —
        the chain is uniformly slowed but not biased.

        In particular, **this function CANNOT be used to soften
        Assumption 6.1** (the FalCom irreducibility condition): 6.1 is
        a property of the graph + candidate set, both fixed across the
        chain run, so the violation is constant. The proposal mechanism
        (``bipartition_tree``) also enforces feasibility upstream — no
        infeasible state is ever proposed for the acceptance step to
        see. To soften 6.1, repair the candidate set before the chain
        (``repair_facility_density``) or penalise *use* of artificial
        candidates via
        :func:`falcomchain.markovchain.energy.compute_artificial_facility_penalty`
        in the energy.

    Use this for genuine per-state constraints, e.g. district
    perimeter caps or contiguity tolerance levels that vary with
    the partition.

    :param violation_fn: Callable mapping a :class:`ChainState` to a
        non-negative violation magnitude. Must be state-dependent.
    :param beta_feas: Inverse "soft-constraint temperature." Higher
        values penalise violations more strongly.
        ``beta_feas = 0`` recovers ``base_accept``.
    :param base_accept: Underlying acceptance rule (default
        :func:`always_accept`). Use :func:`boltzmann` to combine with
        Boltzmann optimisation on top of the soft constraint.
    :returns: An acceptance function ``(proposed, current) -> bool``.
    """
    if beta_feas < 0:
        raise ValueError(f"beta_feas must be non-negative, got {beta_feas}")

    def _accept(proposed: ChainState, current: ChainState) -> bool:
        if not base_accept(proposed, current):
            return False
        v = violation_fn(proposed)
        if v <= 0:
            return True
        if beta_feas == 0:
            return True
        log_p = -beta_feas * v
        return math.log(rng.random()) <= log_p

    return _accept
