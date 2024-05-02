from dataclasses import dataclass, field
from typing import Self, Optional


@dataclass(frozen=True, kw_only=True)
class LayerActionResult:
    """
    Represents the result of a layer action.
    """

    cumulative_score: float = field(default=0.0)
    """The cumulative score of the action"""

    is_valid: bool = field(default=True)
    """Whether the action taken is valid"""

    @property
    def is_scored(self) -> bool:
        """Whether the taken action is scored"""
        return self.cumulative_score is not None

    @classmethod
    def invalid(cls) -> Self:
        """
        Creates an invalid action result.
        :return: invalid action result
        """
        return cls(is_valid=False)
