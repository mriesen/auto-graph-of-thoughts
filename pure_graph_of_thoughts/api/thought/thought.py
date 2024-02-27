from dataclasses import dataclass, field

from .state import State


@dataclass(frozen=True)
class Thought:
    """
    Represents a thought.
    """

    state: State = field(default_factory=lambda: {})
    """The internal state of the thought"""

    score: float | None = field(default=None)
    """The score of the thought if applicable"""
