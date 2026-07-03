"""
Ensemble statistics coordinator.

:class:`EnsembleStats` is the main entry point for ensemble analysis.
It wraps :class:`BoundaryCounter`, :class:`FacilityCounter`, and
:class:`CapacityStats`, handling burn-in and thinning, and exposes a
single :meth:`observe` callback suitable for the ``MarkovChain``
callbacks list.
"""

from .boundary import BoundaryCounter
from .capacity import CapacityStats
from .facility import FacilityCounter


class EnsembleStats:
    """
    Accumulates boundary, facility, and capacity statistics from an
    MCMC chain, with configurable burn-in and thinning.

    **Usage as a live callback** (recommended):

    .. code-block:: python

        ensemble = EnsembleStats(burn_in=100, thin=5)
        chain = MarkovChain(..., callbacks=[ensemble.observe])
        for state in chain:
            pass
        report = ensemble.report()

    **Usage as a post-hoc loop**:

    .. code-block:: python

        ensemble = EnsembleStats(burn_in=100, thin=5)
        for state in chain:
            ensemble.observe(state, accepted=True)
        report = ensemble.report()

    :param burn_in: Number of initial steps to skip.
    :param thin: Only observe every *thin*-th step after burn-in.
    """

    __slots__ = ("boundary", "facility", "capacity", "_burn_in", "_thin", "_step")

    def __init__(self, burn_in=0, thin=1):
        self.boundary = BoundaryCounter()
        self.facility = FacilityCounter()
        self.capacity = CapacityStats()
        self._burn_in = burn_in
        self._thin = max(1, thin)
        self._step = 0

    @property
    def n_samples(self):
        """Number of samples actually recorded (after burn-in and thinning)."""
        return self.boundary.n_samples

    def observe(self, state, accepted):
        """
        Record one step from the chain.

        Skips the first *burn_in* steps, then records every *thin*-th
        step. Pass this method as a callback to :class:`MarkovChain`.

        :param state: Current :class:`~falcomchain.markovchain.ChainState`.
        :param accepted: Whether the proposal was accepted.
        """
        self._step += 1
        if self._step <= self._burn_in:
            return
        if (self._step - self._burn_in - 1) % self._thin != 0:
            return
        self.boundary.observe(state, accepted)
        self.facility.observe(state, accepted)
        self.capacity.observe(state, accepted)

    def report(self):
        """
        Return a summary dict of all ensemble statistics.

        Keys:

        - ``n_samples``: Number of recorded samples.
        - ``boundary_frequencies``: Dict of edge -> frequency.
        - ``facility_frequencies``: Dict of candidate -> frequency.
        - ``super_facility_frequencies``: Dict of candidate -> frequency (level-2).
        - ``essential_facilities``: Candidates in >= 90% of samples.
        - ``substitutable_facilities``: Candidates in < 50% of samples.
        - ``robust_boundaries``: Edges in >= 90% of samples.
        - ``fragile_boundaries``: Edges in < 50% of samples.
        - ``capacity``: Demand CV, max/mean radius summaries.

        :rtype: dict
        """
        return {
            "n_samples": self.n_samples,
            "boundary_frequencies": self.boundary.frequencies(),
            "facility_frequencies": self.facility.frequencies(),
            "super_facility_frequencies": self.facility.super_frequencies(),
            "essential_facilities": self.facility.essential(),
            "substitutable_facilities": self.facility.substitutable(),
            "robust_boundaries": self.boundary.robust(),
            "fragile_boundaries": self.boundary.fragile(),
            "capacity": self.capacity.summary(),
        }
