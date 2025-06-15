from enum import Enum
from typing import Callable, Any, Mapping

from ..language_model import create_simulated_realistic_chat_gpt_sum_list, create_simulated_deterministic_chat_gpt_sum_list
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel

from ..language_model.simulated_chat_gpt_count_keywords import create_simulated_realistic_chat_gpt_count_keywords, \
    create_simulated_deterministic_chat_gpt_count_keywords
from ..language_model.simulated_chat_gpt_intersect_set import create_simulated_realistic_chat_gpt_intersect_set, \
    create_simulated_deterministic_chat_gpt_intersect_set
from ..language_model.simulated_chat_gpt_sort_list import create_simulated_realistic_chat_gpt_sort_list, \
    create_simulated_deterministic_chat_gpt_sort_list
from auto_graph_of_thoughts.experiment.experiment_task_type import ExperimentTaskType


class LanguageModelSimulationType(Enum):
    """
    Represents the type of language model simulation.
    """
    REALISTIC = 'realistic'
    """A realistic simulation of a language model."""

    DETERMINISTIC = 'deterministic'
    """A simplified simulation of a language model."""

    def get_factory_function(self, task_type: ExperimentTaskType) -> Callable[[int, Mapping[str, Any]], SimulatedLanguageModel]:
        """The factory function of a language model simulation type"""

        if task_type == ExperimentTaskType.SUM_LIST:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_sum_list
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_sum_list
            raise

        if task_type == ExperimentTaskType.SORT_LIST:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_sort_list
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_sort_list

        if task_type == ExperimentTaskType.INTERSECT_SET:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_intersect_set
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_intersect_set

        if task_type == ExperimentTaskType.COUNT_KEYWORDS:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_count_keywords
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_count_keywords

        raise


class LanguageModelSimulationTypeException(Exception):
    """
    An exception that is thrown in context of a language model simulation type.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)