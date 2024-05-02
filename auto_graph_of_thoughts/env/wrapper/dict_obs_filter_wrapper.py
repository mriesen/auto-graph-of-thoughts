from typing import Set

from gymnasium.wrappers import FilterObservation

from ..graph_of_thoughts_env import GraphOfThoughtsEnv
from ...obs import ObservationComponent


class DictObsFilterWrapper(FilterObservation):
    """
    A filter wrapper for filtering a dictionary observation.
    """

    _observation_filter: Set[str]

    def __init__(self, env: GraphOfThoughtsEnv, observation_filter: Set[ObservationComponent]) -> None:
        """
        Instantiates a new dictionary observation filter wrapper.
        :param env: environment
        :param observation_filter: observation filter
        """
        super().__init__(env, [
            key.value for key in observation_filter
        ])
