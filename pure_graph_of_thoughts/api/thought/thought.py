from dataclasses import dataclass, field
from typing import Optional

from ..internal.id import Id
from ..state import State


@dataclass(frozen=True, kw_only=True)
class Thought:
    """
    Represents a thought.
    """

    origin_id: Optional[Id] = field(default=None)
    """The ID of the thought's origin"""

    state: State
    """The internal state of the thought"""

    score: Optional[float] = field(default=None)
    """The score of the thought if applicable"""

    cumulative_score: float = field(default=0.0)
    """The cumulative score of the thought"""
