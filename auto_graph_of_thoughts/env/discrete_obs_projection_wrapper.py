from typing import Any, Tuple, Dict, SupportsFloat

import numpy as np
from gymnasium import Wrapper
from gymnasium.vector.utils import spaces

from .graph_of_thoughts_env import GraphOfThoughtsEnv, ObsType, ActType, GraphOfThoughtsEnvException

WrapperObsType = np.int64
WrapperActType = np.int64


class DiscreteObsProjectionWrapper(Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]):
    """
    An observation projection wrapper for projecting a dictionary observation space onto a discrete observation space.
    """

    _observation_projection: str

    def __init__(self, env: GraphOfThoughtsEnv, observation_projection: str):
        """
        Instantiates a new discrete observation projection wrapper.
        :param env: environment
        :param observation_projection: observation projection
        """
        super().__init__(env)
        self._observation_projection = observation_projection
        if not isinstance(env.observation_space, spaces.Dict):
            raise GraphOfThoughtsEnvException('Observation space must be of type Dict')
        self.observation_space = env.observation_space[self._observation_projection]
        self.action_space = env.action_space

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[self._observation_projection], reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return observation[self._observation_projection], info
