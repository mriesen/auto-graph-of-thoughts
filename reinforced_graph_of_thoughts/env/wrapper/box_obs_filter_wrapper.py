import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.vector.utils import spaces

from .env_wrapper_exception import EnvWrapperException
from ..graph_of_thoughts_env import ObsType, ActType
from ...obs import ObservationComponent

WrapperObsType = np.float32
WrapperActType = np.int64


class BoxObsFilterWrapper(ObservationWrapper[WrapperObsType, ActType, ObsType]):
    """
    An observation filter wrapper for creating a box observation from a dictionary observation.
    """

    _observation_filter: str

    def __init__(self, env: Env[ObsType, ActType], observation_filter: ObservationComponent) -> None:
        """
        Instantiates a new discrete observation filter wrapper.
        :param env: environment
        :param observation_filter: observation filter
        """
        super().__init__(env)
        self._observation_filter = observation_filter.value
        if not isinstance(env.observation_space, spaces.Dict):
            raise EnvWrapperException('Observation space must be of type Dict')
        self.observation_space = env.observation_space[self._observation_filter]
        self.action_space = env.action_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        box_observation: WrapperObsType = observation[self._observation_filter]
        return box_observation

