from dataclasses import dataclass, field
from typing import Optional, Self

from .action_type import ActionType
from .layer_action import LayerAction


@dataclass(kw_only=True)
class GraphStepReward:
    """
    Represents a reward for a step in a graph of thoughts environment.
    """

    _is_invalid: bool = field(default=False)
    _is_final: bool = field(default=False)
    _scored: Optional[bool] = field(default=None)

    depth: int = field(default=0)
    """The graph's depth"""

    n_operations: int = field(default=0)
    """The number of executed operations"""

    max_ops: int
    """The maximum number of operations"""

    max_depth: int
    """The maximum depth"""

    action: LayerAction
    """The action taken"""

    @property
    def is_solved(self) -> bool:
        """Whether the problem is solved"""
        return self._is_final and self._scored is not None and self._scored

    def invalid(self) -> Self:
        """Marks the action taken as invalid"""
        self._is_invalid = True
        return self

    def final(self) -> Self:
        """Marks the step as final"""
        self._is_final = True
        return self

    def scored(self, scored: bool) -> Self:
        """
        Sets the achieved score of the action taken.
        :param scored: the achieved score
        :return: reward
        """
        self._scored = scored
        return self

    def __float__(self) -> float:
        return self._calculate_reward() / 100.0

    def _calculate_reward(self) -> float:
        depth_penalty = -(10 / self.max_depth) * self.depth
        if self.action.type == ActionType.Backtrack:
            return -20
        if self._is_invalid:
            if self._is_final:
                return -100
            return -10
        if self._scored is None:
            return 10 + depth_penalty
        if self._scored:
            if self._is_final:
                return 100
            return 10 + depth_penalty
        if self._is_final:
            return -20 + depth_penalty
        return -10 + depth_penalty
