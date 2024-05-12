from typing import Mapping, Tuple, SupportsFloat, Dict, Any

from pure_graph_of_thoughts.api.task import Task
from .action_type import ActionType
from .graph_of_thoughts_env import GraphOfThoughtsEnv, ObsType, GraphOfThoughtsEnvException
from .graph_step_reward_version import GraphStepRewardVersion
from .layer_action import LayerAction
from ..controller import ContinuousGraphController


class OptimalOnlyGraphOfThoughtsEnv(GraphOfThoughtsEnv):
    """
    A graph of thoughts environment with a given optimal path.
    The rewards are shaped strictly according to the optimal path.
    Only the optimal action is rewarded with a positive reward, all other actions are penalized.
    """

    _optimal_path: Mapping[int, LayerAction]

    def __init__(
            self,
            task: Task,
            controller: ContinuousGraphController,
            seed: int,
            max_steps: int,
            reward_version: GraphStepRewardVersion,
            optimal_path: Mapping[int, LayerAction]
    ):
        """
        Instantiates a new optimal only graph of thoughts environment.
        :param task: task to solve
        :param controller: underlying controller
        :param seed: seed for random number generator
        :param reward_version: reward version
        :param max_steps: maximum number of steps per episode
        :param optimal_path: optimal path
        """
        super().__init__(task, controller, seed, reward_version, max_steps)
        self._optimal_path = optimal_path

    def _process_step(self, action: LayerAction) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        optimal_action = self._optimal_path[self.current_depth]
        info: Dict[str, Any] = {}
        reward: float = 0.0

        if self._terminated or self._truncated:
            self._logger.warning('Episode is terminated or truncated, reset environment')
            return self._observation, reward, self._terminated, self._truncated, info

        if self.current_depth > self.max_depth - 1:
            self._truncated = True
        elif action != optimal_action:
            reward = -0.1
        elif action.type == ActionType.APPEND_OPERATION:
            reward = 0.1
            if action.operation is None:
                raise GraphOfThoughtsEnvException('Operation is None')
            result = self._controller.append_layer(action.operation)
        elif action.type == ActionType.STOP:
            self._terminated = True
            reward = 1.0

        return self._observation, reward, self._terminated, self._truncated, info
