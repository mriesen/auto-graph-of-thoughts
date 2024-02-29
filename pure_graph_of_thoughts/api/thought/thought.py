from dataclasses import dataclass, field
from typing import Optional

from ..state import State
from ..graph import Node


@dataclass(frozen=True)
class Thought:
    """
    Represents a thought.
    """

    origin: Optional[Node] = field(default=None)
    """The origin of the thought"""

    state: State = field(default_factory=lambda: {})
    """The internal state of the thought"""

    score: Optional[float] = field(default=None)
    """The score of the thought if applicable"""
