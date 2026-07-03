from typing import Optional

from falcomchain.partition import Partition
from falcomchain.markovchain.facility import FacilityAssignment, SuperFacilityAssignment


class ChainState:
    """
    Wraps a Partition with the MCMC context needed for the
    Metropolis-Hastings acceptance step.

    The proposal step is responsible for setting energy,
    log_proposal_ratio, and feasible before returning a ChainState.
    beta is set once at chain initialisation and copied forward.

    :ivar partition:           The underlying partition state.
    :ivar energy:              E(s) — access energy of this state.
    :ivar log_proposal_ratio:  ``log( q(s|s') / q(s'|s) )`` from the proposal.
    :ivar beta:                Inverse temperature (>= 0).
    :ivar feasible:            R(s') — hard constraint indicator.
    """

    __slots__ = (
        "partition",
        "facility",
        "super_facility",
        "energy",
        "log_proposal_ratio",
        "beta",
        "feasible",
        "energy_fn",
        "super_facility_fn",
        "_recorder",
    )

    def __init__(
        self,
        partition: Partition,
        facility: FacilityAssignment,
        energy: float,
        log_proposal_ratio: float,
        beta: float,
        feasible: bool,
        super_facility: Optional[SuperFacilityAssignment] = None,
        energy_fn=None,
        super_facility_fn=None,
    ) -> None:
        self.partition = partition
        self.facility = facility
        self.super_facility = super_facility
        self.energy = energy
        self.log_proposal_ratio = log_proposal_ratio
        self.beta = beta
        self.feasible = feasible
        self.energy_fn = energy_fn
        self.super_facility_fn = super_facility_fn

    @classmethod
    def initial(
        cls,
        partition: Partition,
        energy: float,
        beta: float,
        energy_fn=None,
        super_facility_fn=None,
    ) -> "ChainState":
        """
        Construct the initial ChainState at the start of the chain.
        The proposal ratio is 0 (log 1) and the state is always feasible.
        FacilityAssignment is computed eagerly for all districts.

        :param partition: The initial partition.
        :param energy:    E(s_0) of the initial partition.
        :param beta:      Inverse temperature.
        :param energy_fn: Optional custom energy function(state) -> float.
            Defaults to ``compute_energy`` (demand-weighted travel time).
        :param super_facility_fn: Optional callable
            ``(state) -> SuperFacilityAssignment`` that produces the level-2
            facility assignment. ``None`` (default) means level-2 facilities
            are not computed; ``state.super_facility`` is ``None`` for the
            entire chain. Pass ``SuperFacilityAssignment.from_state`` to enable
            the default Eq. 18 minimax selector, or any custom callable.
        """
        state = cls.__new__(cls)
        state.partition = partition
        state.log_proposal_ratio = 0.0
        state.beta = beta
        state.feasible = True
        state.energy_fn = energy_fn
        state.super_facility_fn = super_facility_fn
        state._recorder = None
        state.facility = FacilityAssignment.from_state(state)
        state.super_facility = (
            super_facility_fn(state) if super_facility_fn is not None else None
        )
        # Recompute energy using custom function if provided
        if energy_fn is not None:
            state.energy = energy_fn(state)
        else:
            state.energy = energy
        return state

    def next(
        self,
        partition: Partition,
        energy: float,
        log_proposal_ratio: float,
        feasible: bool,
    ) -> "ChainState":
        """
        Construct the proposed ChainState from the current one.
        Carries beta, energy_fn, super_facility_fn forward and incrementally
        updates FacilityAssignment.

        :param partition:          Proposed partition.
        :param energy:             E(s') of the proposed state.
        :param log_proposal_ratio: ``log( q(s|s') / q(s'|s) )``.
        :param feasible:           Whether s' passes all hard constraints.
        """
        state = ChainState.__new__(ChainState)
        state.partition = partition
        state.log_proposal_ratio = log_proposal_ratio
        state.beta = self.beta
        state.feasible = feasible
        state.energy_fn = self.energy_fn
        # Defensive: tolerate parents constructed via __new__ without
        # super_facility_fn set (e.g., in legacy tests).
        super_facility_fn = getattr(self, "super_facility_fn", None)
        state.super_facility_fn = super_facility_fn
        state._recorder = self._recorder
        state.facility = FacilityAssignment.updated(self.facility, state)
        # Incremental L2 update when a parent super_facility exists; falls
        # back to the user-supplied `super_facility_fn` (typically
        # `SuperFacilityAssignment.from_state`) for the first step or
        # custom callables.
        prev_super_facility = getattr(self, "super_facility", None)
        if (
            super_facility_fn is SuperFacilityAssignment.from_state
            and isinstance(prev_super_facility, SuperFacilityAssignment)
        ):
            state.super_facility = SuperFacilityAssignment.updated(
                prev_super_facility, state
            )
        else:
            state.super_facility = (
                super_facility_fn(state) if super_facility_fn is not None else None
            )
        # Use custom energy function if set, otherwise use the provided value
        if self.energy_fn is not None:
            state.energy = self.energy_fn(state)
        else:
            state.energy = energy
        return state

    # ------------------------------------------------------------------ #
    # Convenience pass-throughs so ChainState can be used like Partition  #
    # ------------------------------------------------------------------ #

    @property
    def parts(self):
        return self.partition.parts

    @property
    def assignment(self):
        return self.partition.assignment

    @property
    def graph(self):
        return self.partition.graph

    @property
    def supergraph(self):
        return self.partition.supergraph

    @property
    def teams(self):
        return self.partition.teams

    @property
    def candidates(self):
        return self.partition.candidates

    @property
    def centers(self) -> dict:
        return self.facility.centers

    @property
    def radii(self) -> dict:
        return self.facility.radii

    @property
    def super_centers(self) -> dict:
        if self.super_facility is None:
            return {}
        return self.super_facility.centers

    @property
    def super_radii(self) -> dict:
        if self.super_facility is None:
            return {}
        return self.super_facility.radii

    @property
    def capacity_level(self):
        return self.partition.capacity_level

    @property
    def step(self):
        return self.partition.step

    def __len__(self):
        return len(self.partition)

    def __repr__(self):
        return (
            f"<ChainState step={self.step} "
            f"energy={self.energy:.4f} "
            f"beta={self.beta} "
            f"feasible={self.feasible}>"
        )
