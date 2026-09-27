class BipartitionWarning(UserWarning):
    """
    Generally raised when it is proving difficult to find a balanced cut.
    """

    pass


class ReselectException(Exception):
    """
    Raised when the tree-splitting algorithm is unable to find a
    balanced cut after some maximum number of attempts, but the
    user has allowed the algorithm to reselect the pair of
    districts from parent graph to try and recombine.
    """

    pass


class BalanceError(Exception):
    """Raised when a balanced cut cannot be found."""


class ProposalRejected(RuntimeError):
    """
    Base class for proposal-internal failures that :class:`~falcomchain.markovchain.MarkovChain`
    treats as a *rejected step*: the chain keeps its current state and records
    the cause in ``chain.rejections``.
    """


class CutSearchExhausted(ProposalRejected):
    """
    Raised by :func:`~falcomchain.tree.tree.bipartition_tree` when no admissible
    subtree is found within the spanning-tree retry budget ``M``.

    :ivar level: ``"base"`` (level-1 recursion) or ``"super"`` (supergraph
        recursion) -- which hierarchical level failed.
    :ivar attempts: The retry budget that was exhausted.
    """

    def __init__(self, level: str, attempts: int, message: str = None):
        self.level = level
        self.attempts = attempts
        if message is None:
            message = (
                f"Could not find a possible cut after {attempts} attempts. "
                f"Supergraph = {level == 'super'}."
            )
        super().__init__(message)


class SuperDistrictTooSmall(ProposalRejected):
    """
    Raised by ``hierarchical_recom`` when the lower-level re-partition of the
    selected super-district produces fewer than ``min_districts_super``
    (paper: kappa^2_min) districts, e.g. when two unit-capacity districts are
    re-cut into a single capacity-2 district. Treated as a rejected proposal.
    """


class PopulationBalanceError(ProposalRejected):
    """
    Raised when an extracted district violates the per-team demand window
    after the cut (a safety net; unreachable under the debt-corrected window).
    Treated as a rejected proposal by the chain.
    """
