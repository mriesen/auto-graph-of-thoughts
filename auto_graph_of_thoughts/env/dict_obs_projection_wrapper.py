from typing import Dict, Any, Tuple, SupportsFloat, Set

import numpy as np
from gymnasium import Wrapper
from gymnasium.vector.utils import spaces

from .graph_of_thoughts_env import GraphOfThoughtsEnv, ObsType, ActType, GraphOfThoughtsEnvException

WrapperObsType = Dict[str, Any]
WrapperActType = np.int64


class DictObsProjectionWrapper(Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]):
    """
    A projection wrapper for projecting a dictionary observation onto another dictionary observation.
    """

    _observation_projection: Set[str]

    def __init__(self, env: GraphOfThoughtsEnv, observation_projection: Set[str]):
        """
        Instantiates a new dictionary observation projection wrapper.
        :param env: environment
        :param observation_projection: observation projection
        """
        super().__init__(env)
        self._observation_projection = observation_projection
        if not isinstance(env.observation_space, spaces.Dict):
            raise GraphOfThoughtsEnvException('Observation space must be of type Dict')
        self.observation_space = spaces.Dict({
            key: space for key, space in env.observation_space.items() if key in self._observation_projection
        })
        self.action_space = env.action_space

    def _project_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value for key, value in observation.items() if key in self._observation_projection
        }

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return self._project_observation(observation), reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        return self._project_observation(observation), info
