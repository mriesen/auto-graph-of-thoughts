from dataclasses import dataclass, field
from typing import Optional, Self

from .action_type import ActionType
from .graph_step_reward_version import GraphStepRewardVersion
from .layer_action import LayerAction


@dataclass(kw_only=True)
class GraphStepReward:
    """
    Represents a reward for a step in a graph of thoughts environment.
    """

    _is_invalid: bool = field(default=False)
    """Whether the action is invalid"""

    _is_final: bool = field(default=False)
    """Whether the action is final"""

    _score: Optional[bool] = field(default=None)
    """The score of the action if applicable"""

    prev_scored: Optional[bool] = field(default=None)
    """The score of the previous action if applicable"""

    depth: int = field(default=0)
    """The graph's depth"""

    n_operations: int = field(default=0)
    """The number of executed operations"""

    version: GraphStepRewardVersion
    """The reward version"""

    max_operations: int
    """The maximum number of operations"""

    max_depth: int
    """The maximum depth"""

    action: LayerAction
    """The action taken"""

    @property
    def is_solved(self) -> bool:
        """Whether the problem is solved"""
        return self._is_final and self._score is not None and self._score

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
        self._score = scored
        return self

    def __float__(self) -> float:
        return self._calculate_reward() / 100.0

    def _calculate_reward(self) -> float:
        if self.version == GraphStepRewardVersion.V0:
            return self._calculate_reward_v0()
        if self.version == GraphStepRewardVersion.V1:
            return self._calculate_reward_v1()
        if self.version == GraphStepRewardVersion.V2:
            return self._calculate_reward_v2()
        if self.version == GraphStepRewardVersion.V3:
            return self._calculate_reward_v3()
        if self.version == GraphStepRewardVersion.V4:
            return self._calculate_reward_v4()
        if self.version == GraphStepRewardVersion.V5:
            return self._calculate_reward_v5()
        if self.version == GraphStepRewardVersion.V6:
            return self._calculate_reward_v6()
        if self.version == GraphStepRewardVersion.V7:
            return self._calculate_reward_v7()
        raise GraphStepRewardException(f'Reward version {self.version} is not supported')

    def _calculate_reward_v0(self) -> float:
        """
        Placeholder reward function
        :return: 0 reward
        """
        return 0

    def _calculate_reward_v1(self) -> float:
        """
        Sparse reward function with depth penalty.
        :return: reward
        """
        n_depth_penalty = -(10 / self.max_depth) * self.depth
        if self._score and self._is_final:
            return 100
        return n_depth_penalty

    def _calculate_reward_v2(self) -> float:
        """
        Sparse reward function with depth penalty and invalid signal.
        :return: reward
        """
        n_depth_penalty = -(10 / self.max_depth) * self.depth
        if self._is_invalid:
            return -10
        if self._score and self._is_final:
            return 100
        return -10 + n_depth_penalty

    def _calculate_reward_v3(self) -> float:
        """
        Reward function with intermediate rewards, depth penalty and invalid signal.
        :return: reward
        """
        n_depth_penalty = -(10 / self.max_depth) * self.depth
        if self._is_invalid:
            return -10
        if self._score:
            if self._is_final:
                return 100
            return 10
        if self._score is None:
            return 5 + n_depth_penalty
        return -10 + n_depth_penalty

    def _calculate_reward_v4(self) -> float:
        """
        Reward function with intermediate rewards, depth penalty, invalid signal and backtrack action penalty.
        :return: reward
        """
        depth_penalty = -(10 / self.max_depth) * self.depth
        if self.action.type == ActionType.BACKTRACK:
            return -20
        if self._is_invalid:
            if self._is_final:
                return -100
            return -10
        if self._score is None:
            return 10 + depth_penalty
        if self._score:
            if self._is_final:
                return 100
            return 10 + depth_penalty
        if self._is_final:
            return -20 + depth_penalty
        return -10 + depth_penalty

    def _calculate_reward_v5(self) -> float:
        """
        Reward function with intermediate rewards, depth penalty, invalid signal and complex backtrack action penalty.
        :return: reward
        """
        n_depth_penalty = -(10 / self.max_depth) * self.depth
        if self._is_invalid:
            return -10
        if self.action.type == ActionType.BACKTRACK:
            if self.prev_scored is not None and not self.prev_scored:
                return 5
            return -10
        if self._score is None:
            return 10 + n_depth_penalty
        if self._score:
            if self._is_final:
                return 100
            return 10 + n_depth_penalty
        if self._is_final:
            return -20 + n_depth_penalty
        return -10 + n_depth_penalty

    def _calculate_reward_v6(self) -> float:
        """
        Reward function with intermediate rewards, operation penalty,
        invalid signal and complex backtrack action penalty.
        :return: reward
        """
        n_ops_penalty = -(10 / self.max_operations) * self.n_operations
        if self._is_invalid:
            return -10
        if self.action.type == ActionType.BACKTRACK:
            if self.prev_scored is not None and not self.prev_scored:
                return 5
            return -10
        if self._score is None:
            return 10 + n_ops_penalty
        if self._score:
            if self._is_final:
                return 100
            return 10 + n_ops_penalty
        if self._is_final:
            return -20 + n_ops_penalty
        return -10 + n_ops_penalty


    def _calculate_reward_v7(self) -> float:
        """
        Reward function with intermediate rewards, depth penalty, invalid signal and complex backtrack action penalty.
        :return: reward
        """
        n_depth_penalty = -(10 / self.max_depth) * self.depth
        if self._is_invalid:
            return -10
        if self.action.type == ActionType.BACKTRACK:
            if self.prev_scored is not None and not self.prev_scored:
                return 15
            return -10
        if self._score is None:
            return 10 + n_depth_penalty
        if self._score:
            if self._is_final:
                return 100
            return 10 + n_depth_penalty
        if self._is_final:
            return -20 + n_depth_penalty
        return -10 + n_depth_penalty

class GraphStepRewardException(Exception):
    """
    An exception that is raised in context of the graph step reward.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
