"""
This module provides the MarkovChain class, which is designed to facilitate the creation
and iteration of Markov chains in the context of political redistricting and gerrymandering
analysis. It allows for the exploration of different districting plans based on specified
constraints and acceptance criteria.

Key Components:

- MarkovChain: The main class used for creating and iterating over Markov chain states.
- Validator: A helper class for validating proposed states in the Markov chain. See
  :class:`~falcomchain.constraints.Validator` for more details.


Usage:
The primary use of this module is to create an instance of MarkovChain with appropriate
parameters like proposal function, constraints, acceptance function, and initial state,
and then to iterate through the states of the Markov chain, yielding a new proposal
at each step.

Dependencies:

- typing: Used for type hints.

Last Updated: 11 Jan 2024
"""

from collections import Counter
from typing import Callable, Iterable, Optional, Union

from falcomchain.constraints import Bounds, Validator
from falcomchain.partition import Partition
from falcomchain.tree.errors import (
    CutSearchExhausted,
    PopulationBalanceError,
    SuperDistrictTooSmall,
)

from .state import ChainState


def classify_rejection(exc: BaseException) -> str:
    """
    Map a proposal-internal exception to a rejection-cause label.

    Labels:

    - ``"cut_search_exhausted:base"``  -- the level-1 recursion found no
      admissible cut within its retry budget (typically a stranded
      candidate-free or under-provisioned residual);
    - ``"cut_search_exhausted:super"`` -- the supergraph recursion stranded a
      final supernode against the discrete capacity constraints;
    - ``"super_district_below_kappa"`` -- the base-level re-cut left the
      selected super-district with fewer than ``min_districts_super`` districts;
    - ``"balance_violation"``          -- the safety-net per-team check failed;
    - ``"runtime_error"``              -- any other ``RuntimeError``.
    """
    if isinstance(exc, CutSearchExhausted):
        return f"cut_search_exhausted:{exc.level}"
    if isinstance(exc, SuperDistrictTooSmall):
        return "super_district_below_kappa"
    if isinstance(exc, PopulationBalanceError):
        return "balance_violation"
    return "runtime_error"


class MarkovChain:
    """
    MarkovChain is a class that creates an iterator for iterating over the states
    of a Markov chain run in a gerrymandering analysis context.

    It allows for the generation of a sequence of partitions (states) of a political
    districting plan, where each partition represents a possible state in the Markov chain.

    Example usage:

    .. code-block:: python

        chain = MarkovChain(proposal, constraints, accept, initial_state, total_steps)
        for state in chain:
            # Do whatever you want - print output, compute scores, ...

    """

    def __init__(
        self,
        proposal: Callable,
        constraints: Union[Iterable[Callable], Validator, Iterable[Bounds], Callable],
        accept: Callable,
        initial_state: ChainState,
        total_steps: int,
        recorder=None,
        callbacks=None,
    ) -> None:
        """
        :param proposal: Function proposing the next ChainState from the current one.
        :type proposal: Callable
        :param constraints: A function with signature ``Partition -> bool`` determining whether
            the proposed next state is valid (passes all binary constraints). Usually
            this is a :class:`~falcomchain.constraints.Validator` class instance.
        :type constraints: Union[Iterable[Callable], Validator, Iterable[Bounds], Callable]
        :param accept: Acceptance function with signature ``(proposed: ChainState, current: ChainState) -> bool``.
            Use ``always_accept`` for unconditional sampling (paper default) or
            ``boltzmann`` for energy-biased optimization. ``boltzmann`` is a
            heuristic optimizer, not a true Metropolis-Hastings sampler — see
            its docstring for the caveats.
        :type accept: Callable
        :param initial_state: Initial :class:`~falcomchain.markovchain.ChainState`.
        :type initial_state: ChainState
        :param total_steps: Number of steps to run.
        :type total_steps: int
        :param recorder: Optional :class:`~falcomchain.tree.snapshot.Recorder` for animation output.
        :type recorder: Optional[Recorder]
        :param callbacks: Optional list of callables invoked after each step with
            signature ``(state: ChainState, accepted: bool) -> None``.
            Use this to attach ensemble analysis or custom logging.
        :type callbacks: Optional[list[Callable]]

        :returns: None

        :raises ValueError: If the initial_state is not valid according to the constraints.
        """
        if callable(constraints):
            is_valid = Validator([constraints])
        else:
            is_valid = Validator(constraints)

        if not is_valid(initial_state.partition):
            failed = [
                constraint
                for constraint in is_valid.constraints  # type: ignore
                if not constraint(initial_state.partition)
            ]
            message = (
                "The given initial_state is not valid according to the constraints. "
                "The failed constraints were: " + ",".join([f.__name__ for f in failed])
            )
            raise ValueError(message)

        self.proposal = proposal
        self.is_valid = is_valid
        self.accept = accept
        self.total_steps = total_steps
        self.initial_state = initial_state
        self.state = initial_state
        self.recorder = recorder
        self.callbacks = callbacks or []
        # Rejection accounting: cause label -> count (see classify_rejection),
        # plus the label of the most recent rejection (None after an accept).
        self.rejections = Counter()
        self.last_rejection = None

        # Attach recorder to state so proposal functions can access it
        if recorder is not None:
            self.state._recorder = recorder

    @property
    def constraints(self) -> Validator:
        """
        Read_only alias for the is_valid property.
        Returns the constraints of the Markov chain.

        :returns: The constraints of the Markov chain.
        :rtype: String
        """
        return self.is_valid

    @constraints.setter
    def constraints(
        self,
        constraints: Union[Iterable[Callable], Validator, Iterable[Bounds], Callable],
    ) -> None:
        """
        Setter for the is_valid property.
        Checks if the initial state is valid according to the new constraints.
        being imposed on the Markov chain, and raises a ValueError if the
        initial state is not valid and lists the failed constraints.

        :param constraints: The new constraints to be imposed on the Markov chain.
        :type constraints: Union[Iterable[Callable], Validator, Iterable[Bounds], Callable]

        :returns: None

        :raises ValueError: If the initial_state is not valid according to the new constraints.
        """

        if callable(constraints):
            is_valid = Validator([constraints])
        else:
            is_valid = Validator(constraints)

        if not is_valid(self.initial_state.partition):
            failed = [
                constraint
                for constraint in is_valid.constraints  # type: ignore
                if not constraint(self.initial_state.partition)
            ]
            message = (
                "The given initial_state is not valid according to the new constraints. "
                "The failed constraints were: " + ",".join([f.__name__ for f in failed])
            )
            raise ValueError(message)

        self.is_valid = is_valid

    def __iter__(self) -> "MarkovChain":
        """
        Resets the Markov chain iterator.

        :returns: Returns itself as an iterator object.
        :rtype: MarkovChain
        """
        self.counter = 0
        self.state = self.initial_state
        self.rejections = Counter()
        self.last_rejection = None
        return self

    def __next__(self) -> Optional[ChainState]:
        """
        Advances the Markov chain to the next state.

        Proposes a new ChainState, validates it against the constraints,
        then calls the accept function with ``(proposed, current)`` to
        decide whether to move. Always yields the current state (accepted
        or unchanged) at each step.

        :returns: The current ChainState after this step.
        :rtype: Optional[ChainState]

        :raises StopIteration: If the total number of steps has been reached.
        """
        if self.counter == 0:
            self.counter += 1
            return self.state

        while self.counter < self.total_steps:
            parent_energy = self.state.energy
            accepted = False

            # Drop the grandparent reference to avoid unbounded memory growth
            if self.state.partition is not None:
                self.state.partition.parent = None

            # Proposal can raise RuntimeError when the spanning-tree heuristic
            # exhausts its retry budget on a particular RNG state. Treat that
            # as a rejection rather than killing the chain — the chain stays
            # at the current state for this step. The proposed_next_state may
            # also fail validity / acceptance checks; same outcome.
            try:
                proposed_next_state = self.proposal(self.state)
            except RuntimeError as exc:
                proposed_next_state = None
                self._record_rejection(classify_rejection(exc))

            if proposed_next_state is not None:
                if not self.is_valid(proposed_next_state.partition):
                    self._record_rejection("constraint_failed")
                elif self.accept(proposed_next_state, self.state):
                    self.state = proposed_next_state
                    accepted = True
                    self.last_rejection = None
                else:
                    self._record_rejection("not_accepted")

            if self.recorder is not None:
                self.recorder.record_step(
                    self.state,
                    accepted=accepted,
                    parent_energy=parent_energy,
                )

            for cb in self.callbacks:
                cb(self.state, accepted)

            self.counter += 1
            return self.state
        if self.recorder is not None:
            self.recorder.close()
        raise StopIteration

    def _record_rejection(self, cause: str) -> None:
        self.rejections[cause] += 1
        self.last_rejection = cause

    def rejection_report(self) -> dict:
        """
        Summarize why proposals were rejected so far.

        :returns: ``{"steps": n, "accepted": a, "rejected": r,
            "acceptance_rate": a / n, "causes": {label: count}}`` where
            ``steps`` counts proposal attempts made so far (the initial
            state is not an attempt).
        :rtype: dict
        """
        steps = max(0, getattr(self, "counter", 1) - 1)
        rejected = sum(self.rejections.values())
        accepted = steps - rejected
        return {
            "steps": steps,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": (accepted / steps) if steps else float("nan"),
            "causes": dict(self.rejections),
        }

    def __len__(self) -> int:
        """
        Returns the total number of steps in the Markov chain.

        :returns: The total number of steps in the Markov chain.
        :rtype: int
        """
        return self.total_steps

    def __repr__(self) -> str:
        return "<MarkovChain [{} steps]>".format(len(self))

    def with_progress_bar(self):
        """
        Wraps the Markov chain in a tqdm progress bar.

        Useful for long-running Markov chains where you want to keep track
        of the progress. Requires the `tqdm` package to be installed.

        :returns: A tqdm-wrapped Markov chain.
        """
        from tqdm.auto import tqdm

        return tqdm(self)
