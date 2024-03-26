from dataclasses import dataclass, field
from typing import Optional

from ..graph.operation import OperationNode
from ..state import State


@dataclass(frozen=True, kw_only=True)
class Thought:
    """
    Represents a thought.
    """

    origin: Optional[OperationNode] = field(default=None, repr=False)
    """The origin of the thought"""

    state: State
    """The internal state of the thought"""

    score: Optional[float] = field(default=None)
    """The score of the thought if applicable"""
