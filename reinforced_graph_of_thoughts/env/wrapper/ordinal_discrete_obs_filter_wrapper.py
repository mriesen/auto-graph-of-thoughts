from gymnasium import ObservationWrapper, Env
from gymnasium.vector.utils import spaces

from ..graph_of_thoughts_env import ObsType, ActType, GraphOfThoughtsEnvException
from ...obs import ObservationComponent
from ...space.ordinal_discrete_space import ORDINAL_DISCRETE_TYPE

WrapperObsType = ORDINAL_DISCRETE_TYPE


class OrdinalDiscreteObsFilterWrapper(ObservationWrapper[WrapperObsType, ActType, ObsType]):
    """
    An observation filter wrapper for creating an ordinal discrete observation out of a dictionary observation.
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
            raise GraphOfThoughtsEnvException('Observation space must be of type Dict')
        self.observation_space = env.observation_space[self._observation_filter]
        self.action_space = env.action_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        ordinal_discrete_observation: WrapperObsType = observation[self._observation_filter]
        return ordinal_discrete_observation
