from dataclasses import dataclass, field
from typing import Optional

from ..state import State
from ..graph.operation import OperationNode


@dataclass(frozen=True)
class Thought:
    """
    Represents a thought.
    """

    origin: Optional[OperationNode] = field(default=None)
    """The origin of the thought"""

    state: State = field(default_factory=lambda: {})
    """The internal state of the thought"""

    score: Optional[float] = field(default=None)
    """The score of the thought if applicable"""
