"""
Facility assignment stability analysis across an MCMC ensemble.

Counts how often each candidate node is selected as a facility center
across samples. Candidates appearing in >90% of samples are
*essential*; those in <50% are *substitutable*.
"""


class FacilityCounter:
    """
    Counts per-candidate facility selection frequency across MCMC samples.

    Tracks both level-1 (district) and level-2 (superdistrict) facility
    assignments. On rejected steps the previous facility set is re-counted.

    :ivar n_samples: Number of samples observed so far.
    """

    __slots__ = (
        "_counts",
        "_super_counts",
        "_n_samples",
        "_current_centers",
        "_current_super_centers",
    )

    def __init__(self):
        self._counts = {}               # candidate node -> int
        self._super_counts = {}         # candidate node -> int (level-2)
        self._n_samples = 0
        self._current_centers = set()
        self._current_super_centers = set()

    @property
    def n_samples(self):
        return self._n_samples

    def observe(self, state, accepted):
        """
        Record one sample from the chain.

        :param state: Current :class:`~falcomchain.markovchain.ChainState`.
        :param accepted: Whether the proposal was accepted this step.
        """
        if accepted or self._n_samples == 0:
            self._current_centers = set(state.facility.centers.values())
            if state.super_facility is not None:
                self._current_super_centers = set(
                    state.super_facility.centers.values()
                )
            else:
                self._current_super_centers = set()

        for c in self._current_centers:
            self._counts[c] = self._counts.get(c, 0) + 1
        for c in self._current_super_centers:
            self._super_counts[c] = self._super_counts.get(c, 0) + 1
        self._n_samples += 1

    def frequencies(self):
        """
        Return per-candidate facility frequency in [0, 1].

        :returns: Dict mapping candidate node -> frequency.
        :rtype: dict
        """
        if self._n_samples == 0:
            return {}
        return {node: c / self._n_samples for node, c in self._counts.items()}

    def super_frequencies(self):
        """
        Return per-candidate level-2 facility frequency in [0, 1].

        :returns: Dict mapping candidate node -> frequency.
        :rtype: dict
        """
        if self._n_samples == 0:
            return {}
        return {
            node: c / self._n_samples for node, c in self._super_counts.items()
        }

    def essential(self, threshold=0.9):
        """Candidates selected as facilities in >= *threshold* fraction of samples."""
        return {n: f for n, f in self.frequencies().items() if f >= threshold}

    def substitutable(self, threshold=0.5):
        """Candidates selected as facilities in < *threshold* fraction of samples."""
        return {n: f for n, f in self.frequencies().items() if f < threshold}
