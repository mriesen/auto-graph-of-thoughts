from enum import Enum
from typing import Dict, Any, Self


class ObservationComponent(Enum):
    """
    Represents a component of an observation.
    """

    depth = 'depth'
    breadth = 'breadth'
    complexity = 'complexity'
    local_complexity = 'local_complexity'
    prev_action = 'prev_action'
    prev_scored = 'prev_scored'
    prev_scorable = 'prev_scorable'
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
