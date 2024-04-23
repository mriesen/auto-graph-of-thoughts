from enum import Enum
from typing import Dict, Any, Self, Set


class ObservationComponent(Enum):
    """
    Represents a component of an observation.
    """

    depth = 'depth'
    breadth = 'breadth'
    complexity = 'complexity'
    local_complexity = 'local_complexity'
    prev_action = 'prev_action'
    prev_score = 'prev_score'
    divergence = 'divergence'

    @classmethod
    def create_dict(cls, observation_dictionary: Dict[Self, Any]) -> Dict[str, Any]:
        """
        Creates a proper string-indexed dictionary out of a dictionary with observation components as keys.
        :param observation_dictionary: observation dictionary
        :return: dictionary with strings as keys
        """
        return {
            key.value: value for key, value in observation_dictionary.items()
        }

    @classmethod
    def create_set(cls, observation_set: Set[Self]) -> Set[str]:
        """
        Creates a proper string set out of an observation component set.
        :param observation_set: observation set
        :return: set with strings
        """
        return {
            component.value for component in observation_set
        }
