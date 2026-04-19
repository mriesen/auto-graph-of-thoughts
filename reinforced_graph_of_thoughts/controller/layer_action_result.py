from dataclasses import dataclass, field
from typing import Self, Optional


@dataclass(frozen=True, kw_only=True)
class LayerActionResult:
    """
    Represents the result of a layer action.
    """

    score: Optional[float] = field(default=None)
    """The score of the action"""

    is_valid: bool = field(default=True)
    """Whether the action taken is valid"""

    @property
    def is_scored(self) -> bool:
        """Whether the taken action is scored"""
        return self.score is not None

    @classmethod
    def invalid(cls) -> Self:
        """
        Creates an invalid action result.
        :return: invalid action result
        """
        return cls(is_valid=False)
