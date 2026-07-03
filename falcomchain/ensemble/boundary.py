"""
Boundary frequency analysis across an MCMC ensemble.

Counts how often each edge in the graph serves as a district boundary
across the collected samples. Edges with high frequency (>90%) are
*robust* boundaries; low frequency (<50%) are *fragile*.
"""


class BoundaryCounter:
    """
    Counts per-edge boundary frequency across MCMC samples.

    On each observation, edges whose endpoints belong to different
    districts are counted as boundary edges. On rejected steps the
    previous boundary set is re-counted (correct for MCMC ergodic
    averages, where time spent in a state matters).

    :ivar n_samples: Number of samples observed so far.
    """

    __slots__ = ("_counts", "_n_samples", "_current_boundaries")

    def __init__(self):
        self._counts = {}               # edge (canonical) -> int
        self._n_samples = 0
        self._current_boundaries = set()  # set of canonical edge tuples

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
            self._current_boundaries = _boundary_edges(state.partition)

        for edge in self._current_boundaries:
            self._counts[edge] = self._counts.get(edge, 0) + 1
        self._n_samples += 1

    def frequencies(self):
        """
        Return per-edge boundary frequency in [0, 1].

        :returns: Dict mapping ``(u, v)`` -> frequency.
        :rtype: dict
        """
        if self._n_samples == 0:
            return {}
        return {edge: c / self._n_samples for edge, c in self._counts.items()}

    def robust(self, threshold=0.9):
        """Edges that are boundaries in >= *threshold* fraction of samples."""
        return {e: f for e, f in self.frequencies().items() if f >= threshold}

    def fragile(self, threshold=0.5):
        """Edges that are boundaries in < *threshold* fraction of samples."""
        return {e: f for e, f in self.frequencies().items() if f < threshold}


def _boundary_edges(partition):
    """Return the set of canonical boundary edges for a partition."""
    mapping = partition.assignment.mapping
    boundaries = set()
    for u, v in partition.graph.edges():
        if mapping[u] != mapping[v]:
            boundaries.add((min(u, v), max(u, v)) if u < v else (u, v))
    return boundaries
