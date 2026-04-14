import math

from falcomchain.markovchain.state import ChainState
from falcomchain.random import rng


def always_accept(_proposed: ChainState, _current: ChainState) -> bool:
    """
    Acceptance function that accepts every proposed state unconditionally.
    Use this to run the Markov chain as a sampler without any rejection step.

    :param proposed: The proposed chain state.
    :param current: The current chain state (unused).
    :returns: Always ``True``.
    :rtype: bool
    """
    return True


def metropolis_hastings(state: ChainState, parent: ChainState) -> bool:
    """
    Metropolis-Hastings acceptance rule.

    Accepts the proposed state with probability

        alpha(s -> s') = min(1, R(s') * exp(-beta * (E(s') - E(s))) * q(s|s') / q(s'|s))

    where:
      - R(s') is the hard feasibility indicator  → state.feasible
      - E(s') is the energy of the proposed state → state.energy
      - E(s)  is the energy of the current state  → parent.energy
      - log(q(s|s') / q(s'|s)) is the log proposal ratio → state.log_proposal_ratio
      - beta is the inverse temperature            → state.beta

    :param state:  The proposed ChainState.
    :param parent: The current ChainState.
    :returns: True if the proposal is accepted, False otherwise.
    :rtype: bool
    """
    if not state.feasible:
        return False

    if state.beta == 0:
        return True

    delta_energy = state.energy - parent.energy
    log_alpha = -state.beta * delta_energy + state.log_proposal_ratio

    return math.log(rng.random()) <= log_alpha
