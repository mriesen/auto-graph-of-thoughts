import logging
from typing import Any, SupportsFloat, Sequence, Tuple, Dict, Optional

import numpy as np
from gymnasium import Env
from gymnasium.vector.utils import spaces

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.graph.thought import GraphOfThoughts
from pure_graph_of_thoughts.api.operation import Operation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, InvertedOperationIndex
from .action_type import ActionType
from .graph_step_reward import GraphStepReward
from .layer_action import LayerAction
from .observation_component import ObservationComponent
from ..controller import ContinuousGraphController, LayerActionResult

ObsType = Dict[str, np.int64]
ActType = np.int64

OPTIONAL_BOOL_REPRESENTATION = 3
ABSENT_BOOL = 2


def optional_bool_to_int(value: Optional[bool]) -> int:
    if value is None:
        return ABSENT_BOOL
    return int(value)


class GraphOfThoughtsEnv(Env[ObsType, ActType]):
    """
    The graph of thoughts environment.
    """

    _task: Task
    _controller: ContinuousGraphController
    _max_steps: int

    _terminated: bool
    _truncated: bool
    _total_reward: float
    _n_steps: int
    _prev_action: Optional[LayerAction]
    _prev_result: Optional[LayerActionResult]

    _logger: logging.Logger

    @property
    def max_depth(self) -> int:
        """The maximum depth"""
        return self._controller.max_depth

    @property
    def max_breadth(self) -> int:
        """The maximum breadth"""
        return self._controller.max_breadth

    @property
    def current_depth(self) -> int:
        """The current depth"""
        return self._controller.current_depth

    @property
    def current_breadth(self) -> int:
        """The current breadth"""
        return self._controller.current_breadth

    @property
    def max_complexity(self) -> int:
        """The maximum complexity"""
        return self._controller.max_complexity

    @property
    def n_operations(self) -> int:
        """The number of executed operations"""
        return self._controller.n_operations

    @property
    def graph_of_thoughts(self) -> Optional[GraphOfThoughts]:
        """The graph of thoughts"""
        return self._controller.graph_of_thoughts

    @property
    def graph_of_operations(self) -> GraphOfOperations:
        """The graph of operations"""
        return self._controller.graph_of_operations

    @property
    def _optional_action_representation(self) -> int:
        return len([ActionType.Stop, ActionType.Backtrack]) + len(self._task.operations) + 1

    @property
    def _prev_score(self) -> Optional[bool]:
        return self._prev_result.score == 1.0 if (
                self._prev_result is not None and self._prev_result.is_scored
        ) else None

    @property
    def _prev_scorable(self) -> bool:
        return self._prev_result.is_scored if self._prev_result is not None else False

    @property
    def _observation(self) -> ObsType:
        return ObservationComponent.create_dict({
            ObservationComponent.depth: np.int64(self.current_depth),
            ObservationComponent.breadth: np.int64(self.current_breadth),
            ObservationComponent.divergence: np.array([int(self._controller.divergence)], dtype=np.int8),
            ObservationComponent.complexity: np.int64(self._controller.complexity),
            ObservationComponent.local_complexity: np.int64(self._controller.local_complexity),
            ObservationComponent.prev_action: np.int64(self.encode_optional_action(self._prev_action)),
            ObservationComponent.prev_score: np.int64(optional_bool_to_int(self._prev_score))
        })

    def __init__(
            self,
            task: Task,
            controller: ContinuousGraphController,
            seed: int,
            max_steps: int
    ) -> None:
        """
        Instantiates a new graph of thoughts environment.

        :param task: task to solve
        :param controller: underlying controller
        :param seed: seed for the random number generator
        :param max_steps: maximum number of steps per episode
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        self._task = task
        self._controller = controller
        self._max_steps = max_steps

        self._terminated = False
        self._truncated = False
        self._total_reward = 0.0
        self._n_steps = 0
        self._prev_action = None
        self._prev_result = None

        n_operations: int = len(self._task.operations)
        depth_representation: int = self.max_depth + 1
        breadth_representation: int = self.max_breadth + 1
        action_representation: int = len([ActionType.Stop, ActionType.Backtrack]) + n_operations
        complexity_representation: int = self.max_complexity + 1
        self.observation_space = spaces.Dict(ObservationComponent.create_dict({
            ObservationComponent.depth: spaces.Discrete(depth_representation, seed=seed),
            ObservationComponent.breadth: spaces.Discrete(breadth_representation, seed=seed),
            ObservationComponent.divergence: spaces.MultiBinary(n=1, seed=seed),
            ObservationComponent.complexity: spaces.Discrete(complexity_representation, seed=seed),
            ObservationComponent.local_complexity: spaces.Discrete(complexity_representation, seed=seed),
            ObservationComponent.prev_action: spaces.Discrete(self._optional_action_representation, seed=seed),
            ObservationComponent.prev_score: spaces.Discrete(OPTIONAL_BOOL_REPRESENTATION, seed=seed)
        }), seed=seed)
        self.action_space = spaces.Discrete(action_representation, seed=seed)

    def step(self, encoded_action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if self._terminated or self._truncated:
            self._logger.warning('Episode is terminated or truncated, reset environment')
            return self._observation, 0.0, self._terminated, self._truncated, {}

        if self._n_steps >= self._max_steps:
            self._truncated = True
            return self._observation, 0.0, self._terminated, self._truncated, {}

        self._n_steps += 1

        action = self.decode_action(encoded_action)
        return self._process_step(action)

    def reset(
            self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._terminated = False
        self._truncated = False
        self._total_reward = 0.0
        self._n_steps = 0
        self._prev_action = None
        self._prev_result = None
        self._controller.reset()

        if self._observation not in self.observation_space:
            print(self._observation)

        return self._observation, {}

    def _process_step(self, action: LayerAction) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        reward = GraphStepReward(action=action, max_depth=self.max_depth, max_ops=self.max_depth * self.max_breadth)

        self._prev_action = action

        if action.type == ActionType.Stop:
            self._terminated = True
            reward = self._calculate_final_reward(reward)
            info['solved'] = reward.is_solved
            return self._observation, reward, self._terminated, self._truncated, info

        result: Optional[LayerActionResult] = None
        if action.type == ActionType.Backtrack:
            result = self._controller.remove_sink_layer()
        elif action.type == ActionType.AppendOperation:
            if action.operation is None:
                raise GraphOfThoughtsEnvException('Operation to append is None')
            result = self._controller.append_layer(action.operation)

        reward.depth = self.current_depth
        reward.n_operations = self.n_operations

        if result is None:
            raise GraphOfThoughtsEnvException('Result is None')

        self._prev_result = result

        if not result.is_valid:
            reward = reward.invalid()
        elif result.is_scored:
            reward = reward.scored(result.score == 1.0)

        if self._observation not in self.observation_space:
            print(self._observation)
        return self._observation, reward, self._terminated, self._truncated, info

    def _calculate_final_reward(self, reward: GraphStepReward) -> GraphStepReward:
        reward = reward.final()
        graph_of_thoughts: Optional[GraphOfThoughts] = self._controller.graph_of_thoughts
        if graph_of_thoughts is None:
            return reward.invalid()
        sink_thoughts = [
            thought_node.thought for thought_node in graph_of_thoughts.layers[-1]
        ]
        if len(sink_thoughts) > 1:
            return reward.scored(False)
        init_state: State = self._controller.init_state
        final_state: State = sink_thoughts[0].state
        is_solved = self._task.evaluator.evaluate(init_state, final_state)
        if is_solved:
            return reward.scored(True)
        else:
            return reward.scored(False)

    def decode_action(self, encoded_action: np.int64) -> LayerAction:
        operations: Sequence[Operation] = self._task.operations
        scalar = encoded_action.item()
        if scalar <= 1:
            return LayerAction(ActionType(scalar))
        operation_index = scalar - ActionType.AppendOperation.value
        return LayerAction(type=ActionType.AppendOperation, operation=operations[operation_index])

    def encode_action(self, action: LayerAction) -> np.int64:
        inverted_operation_index: InvertedOperationIndex = self._task.inverted_operation_index
        if action.type != ActionType.AppendOperation:
            return np.int64(action.type.value)
        if action.operation is None:
            raise GraphOfThoughtsEnvException('Operation is None')
        return np.int64(ActionType.AppendOperation.value + inverted_operation_index[action.operation.key])

    def encode_optional_action(self, action: Optional[LayerAction]) -> np.int64:
        if action is None:
            return np.int64(self._optional_action_representation - 1)
        return self.encode_action(action)


class GraphOfThoughtsEnvException(Exception):
    """
    Exception raised in the context of a graph of thoughts RL environment.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
