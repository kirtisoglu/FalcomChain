"""
Capacity and demand utilization statistics across an MCMC ensemble.

Tracks per-sample summary statistics (coefficient of variation of
demand-per-team, covering radii) using Welford's online algorithm
for numerically stable streaming mean and variance.
"""

import math


class WelfordStats:
    """
    Online computation of mean and variance via Welford's algorithm.

    Numerically stable for large sample counts. Supports streaming
    updates without storing individual values.
    """

    __slots__ = ("_n", "_mean", "_m2", "_min", "_max")

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    @property
    def std(self):
        return math.sqrt(self.variance)

    @property
    def min(self):
        return self._min if self._n > 0 else None

    @property
    def max(self):
        return self._max if self._n > 0 else None

    def update(self, x):
        """Add a new observation."""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    def summary(self):
        """Return a dict summarizing the accumulated statistics."""
        return {
            "n": self._n,
            "mean": self._mean,
            "std": self.std,
            "min": self._min if self._n > 0 else None,
            "max": self._max if self._n > 0 else None,
        }


class CapacityStats:
    """
    Tracks demand-per-team balance and covering radius distributions
    across MCMC samples.

    For each sample, computes:

    - **Demand CV**: Coefficient of variation of demand-per-team across
      districts within that sample. Low CV = well-balanced workload.
    - **Max radius**: Worst-case covering radius across all districts.
    - **Mean radius**: Average covering radius across all districts.

    These per-sample summaries are accumulated with Welford's algorithm,
    yielding the distribution of each metric over the ensemble.

    :ivar demand_cv: Welford stats for per-sample demand CV.
    :ivar max_radius: Welford stats for per-sample max covering radius.
    :ivar mean_radius: Welford stats for per-sample mean covering radius.
    """

    __slots__ = (
        "demand_cv",
        "max_radius",
        "mean_radius",
        "_n_samples",
        "_last_cv",
        "_last_max_r",
        "_last_mean_r",
    )

    def __init__(self):
        self.demand_cv = WelfordStats()
        self.max_radius = WelfordStats()
        self.mean_radius = WelfordStats()
        self._n_samples = 0
        self._last_cv = 0.0
        self._last_max_r = 0.0
        self._last_mean_r = 0.0

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
            self._recompute(state)

        self.demand_cv.update(self._last_cv)
        self.max_radius.update(self._last_max_r)
        self.mean_radius.update(self._last_mean_r)
        self._n_samples += 1

    def _recompute(self, state):
        """Recompute cached per-sample statistics from the current state."""
        parts = state.partition.parts
        teams = state.partition.teams
        graph = state.partition.graph

        # Demand per team for each district
        demands_per_team = []
        for part_id, nodes in parts.items():
            total_demand = sum(graph.nodes[n]["demand"] for n in nodes)
            n_teams = teams[part_id]
            demands_per_team.append(total_demand / n_teams if n_teams > 0 else 0.0)

        if demands_per_team:
            mean_dpt = sum(demands_per_team) / len(demands_per_team)
            if mean_dpt > 0:
                var = sum((d - mean_dpt) ** 2 for d in demands_per_team) / len(
                    demands_per_team
                )
                self._last_cv = math.sqrt(var) / mean_dpt
            else:
                self._last_cv = 0.0
        else:
            self._last_cv = 0.0

        # Covering radii
        radii = list(state.facility.radii.values())
        if radii:
            self._last_max_r = max(radii)
            self._last_mean_r = sum(radii) / len(radii)
        else:
            self._last_max_r = 0.0
            self._last_mean_r = 0.0

    def summary(self):
        """Return a dict summarizing the accumulated statistics."""
        return {
            "demand_cv": self.demand_cv.summary(),
            "max_radius": self.max_radius.summary(),
            "mean_radius": self.mean_radius.summary(),
        }
