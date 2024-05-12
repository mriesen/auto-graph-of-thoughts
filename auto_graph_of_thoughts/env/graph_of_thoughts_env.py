import logging
from typing import Any, SupportsFloat, Sequence, Tuple, Dict, Optional, Callable, Mapping, List

import numpy as np
from gymnasium import Env
from gymnasium.vector.utils import spaces

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.graph.thought import GraphOfThoughts
from pure_graph_of_thoughts.api.operation import Operation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, InvertedOperationIndex
from .action_type import ActionType
from .graph_observation_component import GraphObservationComponent
from .graph_step_reward import GraphStepReward
from .graph_step_reward_version import GraphStepRewardVersion
from .layer_action import LayerAction
from ..controller import ContinuousGraphController, LayerActionResult
from ..obs import ObservationComponent
from ..space import MultiSpace, OrdinalDiscreteSpace, OptionalBoolSpace, MultiDiscreteSpace

ObsType = Mapping[str, Any]
ActType = np.int64

DEFAULT_ACTION_LOOKBACK = 4
DEFAULT_MAX_STEPS = 100


class GraphOfThoughtsEnv(Env[ObsType, ActType]):
    """
    The graph of thoughts environment.
    """

    _task: Task
    _controller: ContinuousGraphController
    _max_steps: int
    _reward_version: GraphStepRewardVersion
    _transform_observation: Callable[[Mapping[ObservationComponent, Any]], Mapping[str, Any]]

    _terminated: bool
    _truncated: bool
    _total_reward: float
    _n_steps: int
    _action_lookback: int
    _prev_actions: List[Optional[LayerAction]]
    _graph_of_operations_representation: Sequence[Operation]
    _prev_result: Optional[LayerActionResult]
    _is_solved: bool

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
    def max_operations(self) -> int:
        """The maximum number of operations"""
        return self._controller.max_operations

    @property
    def n_operations(self) -> int:
        """The number of executed operations"""
        return self._controller.n_operations

    @property
    def complexity(self) -> int:
        """The complexity of the current task"""
        return self._controller.complexity

    @property
    def local_complexity(self) -> int:
        """The local complexity"""
        return self._controller.local_complexity

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def graph_of_thoughts(self) -> Optional[GraphOfThoughts]:
        """The graph of thoughts"""
        return self._controller.graph_of_thoughts

    @property
    def graph_of_operations(self) -> GraphOfOperations:
        """The graph of operations"""
        return self._controller.graph_of_operations

    @property
    def _operation_representation(self) -> int:
        return len(self._task.operations)

    @property
    def _optional_operation_representation(self) -> int:
        return self._operation_representation + 1

    @property
    def _optional_action_representation(self) -> int:
        return len([ActionType.STOP, ActionType.BACKTRACK]) + len(self._task.operations) + 1

    @property
    def _prev_score(self) -> Optional[bool]:
        return self._prev_result.score == 1.0 if (
                self._prev_result is not None and self._prev_result.is_scored
        ) else None

    @property
    def _prev_scorable(self) -> bool:
        return self._prev_result.is_scored if self._prev_result is not None else False

    @property
    def _all_actions(self) -> Sequence[LayerAction]:
        return [
            LayerAction(ActionType.STOP),
            LayerAction(ActionType.BACKTRACK),
        ] + [
            LayerAction(ActionType.APPEND_OPERATION, operation)
            for operation in self._task.operations
        ]

    @property
    def _graph_operations(self) -> Sequence[Optional[Operation]]:
        operations = [
            layer[0].operation
            for layer in self.graph_of_operations.layers
        ] if self._controller.is_initialized else []
        empty_layers = self.max_depth - len(operations)
        return operations + ([None] * empty_layers)

    @property
    def _observation(self) -> ObsType:
        return self._transform_observation({
            GraphObservationComponent.DEPTH: self.current_depth,
            GraphObservationComponent.BREADTH: self.current_breadth,
            GraphObservationComponent.COMPLEXITY: self._controller.complexity,
            GraphObservationComponent.LOCAL_COMPLEXITY: self._controller.local_complexity,
            GraphObservationComponent.GRAPH_OPERATIONS: [
                self.encode_optional_operation(operation) for operation in self._graph_operations
            ],
            GraphObservationComponent.PREV_ACTIONS: [
                self.encode_optional_action(prev_action)
                for prev_action in self._prev_actions[-self._action_lookback:]
            ],
            GraphObservationComponent.PREV_SCORE: self._prev_score
        })

    def __init__(
            self,
            task: Task,
            controller: ContinuousGraphController,
            seed: int,
            reward_version: GraphStepRewardVersion,
            action_lookback: int = DEFAULT_ACTION_LOOKBACK,
            max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        """
        Instantiates a new graph of thoughts environment.

        :param task: task to solve
        :param controller: underlying controller
        :param seed: seed for the random number generator
        :param reward_version: reward version
        :param action_lookback: the lookback for actions
        :param max_steps: maximum number of steps per episode
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        self._task = task
        self._controller = controller
        self._action_lookback = action_lookback
        self._max_steps = max_steps
        self._reward_version = reward_version

        self._terminated = False
        self._truncated = False
        self._total_reward = 0.0
        self._n_steps = 0
        self._prev_actions = [None] * self._action_lookback
        self._prev_result = None
        self._is_solved = False

        n_operations: int = len(self._task.operations)
        depth_representation: int = self.max_depth + 1
        breadth_representation: int = self.max_breadth + 1
        action_representation: int = len([ActionType.STOP, ActionType.BACKTRACK]) + n_operations
        observation_space = MultiSpace.of({
            GraphObservationComponent.DEPTH: OrdinalDiscreteSpace(depth_representation, seed=seed),
            GraphObservationComponent.BREADTH: OrdinalDiscreteSpace(breadth_representation, seed=seed),
            GraphObservationComponent.COMPLEXITY: OrdinalDiscreteSpace(self.max_complexity, start=1, seed=seed),
            GraphObservationComponent.LOCAL_COMPLEXITY: OrdinalDiscreteSpace(self.max_complexity + 1, start=0,
                                                                             seed=seed),
            GraphObservationComponent.GRAPH_OPERATIONS: MultiDiscreteSpace(
                    [
                        self._optional_operation_representation
                        for _ in range(self.max_depth)
                    ],
                    seed=seed
            ),
            GraphObservationComponent.PREV_ACTIONS: MultiDiscreteSpace(
                    [
                        self._optional_action_representation
                        for _ in range(self._action_lookback)
                    ],
                    seed=seed
            ),
            GraphObservationComponent.PREV_SCORE: OptionalBoolSpace(seed=seed)
        }, seed=seed)
        self.observation_space = observation_space
        self._transform_observation = observation_space.transform
        self.action_space = spaces.Discrete(action_representation, seed=seed)

    def step(self, encoded_action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

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
        self._prev_actions = [None] * self._action_lookback
        self._prev_result = None
        self._is_solved = False
        self._controller.reset()

        if self._observation not in self.observation_space:
            raise GraphOfThoughtsEnvException(f'Observation is not in observation space: {self._observation}')

        return self._observation, {}

    def _process_step(self, action: LayerAction) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        reward = GraphStepReward(
                version=self._reward_version,
                action=action,
                max_depth=self.max_depth,
                max_operations=self.max_operations,
                prev_scored=self._prev_score
        )

        self._prev_actions.append(action)

        if action.type == ActionType.STOP:
            self._terminated = True
            reward = self._calculate_final_reward(reward)
            info['solved'] = reward.is_solved
            self._is_solved = reward.is_solved
            return self._observation, reward, self._terminated, self._truncated, info

        result: Optional[LayerActionResult] = None
        if action.type == ActionType.BACKTRACK:
            result = self._controller.remove_sink_layer()
        elif action.type == ActionType.APPEND_OPERATION:
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
            raise GraphOfThoughtsEnvException(f'Observation is not in observation space: {self._observation}')
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

    def encode_operation(self, operation: Operation) -> int:
        """
        Encodes a given operation
        :param operation: operation to encode
        :return: encoded operation
        """
        inverted_operation_index: InvertedOperationIndex = self._task.inverted_operation_index
        return inverted_operation_index[operation.key]

    def decode_operation(self, encoded_operation: int) -> Operation:
        """
        Decodes an encoded operation.
        :param encoded_operation: encoded operation
        :return: decoded operation
        """
        operations: Sequence[Operation] = self._task.operations
        return operations[encoded_operation]

    def decode_action(self, encoded_action: np.int64) -> LayerAction:
        """
        Decodes an encoded action
        :param encoded_action: action to decode
        :return: decoded action
        """
        scalar = encoded_action.item()
        if scalar <= 1:
            return LayerAction(ActionType(scalar))
        operation_index = scalar - ActionType.APPEND_OPERATION.value
        return LayerAction(type=ActionType.APPEND_OPERATION, operation=self.decode_operation(operation_index))

    def encode_action(self, action: LayerAction) -> int:
        """
        Encodes a given action.
        :param action: action to encode
        :return: encoded action
        """
        if action.type != ActionType.APPEND_OPERATION:
            return int(action.type.value)
        if action.operation is None:
            raise GraphOfThoughtsEnvException('Operation is None')
        return ActionType.APPEND_OPERATION.value + self.encode_operation(action.operation)

    def encode_optional_action(self, action: Optional[LayerAction]) -> int:
        """
        Encodes an optional action.
        :param action: optional action to encode
        :return: encoded optional action
        """
        if action is None:
            return self._optional_action_representation - 1
        return self.encode_action(action)

    def encode_optional_operation(self, operation: Optional[Operation]) -> int:
        """
        Encodes an optional operation
        :param operation: optional operation to encode
        :return: encoded optional operation
        """
        if operation is None:
            return self._optional_operation_representation - 1
        return self.encode_operation(operation)

    def validate_action(self, action: LayerAction) -> bool:
        """
        Validates a given action.
        :param action: action to validate
        :return: whether the action is valid
        """
        if action.type == ActionType.STOP:
            return True
        if action.type == ActionType.BACKTRACK:
            return self._controller.is_initialized
        if action.type == ActionType.APPEND_OPERATION and action.operation is not None:
            return self._controller.validate_append_operation(action.operation)
        return False


class GraphOfThoughtsEnvException(Exception):
    """
    Exception raised in the context of a graph of thoughts RL environment.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
