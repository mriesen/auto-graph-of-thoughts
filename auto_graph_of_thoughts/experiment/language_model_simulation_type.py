from enum import Enum
from typing import Callable

from pure_graph_of_thoughts.api.task import Task

from ..language_model import create_simulated_realistic_chat_gpt_sum_list, create_simulated_deterministic_chat_gpt_sum_list
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel

from ..language_model.simulated_chat_gpt_count_keywords import create_simulated_realistic_chat_gpt_count_keywords, \
    create_simulated_deterministic_chat_gpt_count_keywords
from ..language_model.simulated_chat_gpt_intersect_set import create_simulated_realistic_chat_gpt_intersect_set, \
    create_simulated_deterministic_chat_gpt_intersect_set
from ..language_model.simulated_chat_gpt_sort_list import create_simulated_realistic_chat_gpt_sort_list, \
    create_simulated_deterministic_chat_gpt_sort_list
from ..tasks.count_keywords import count_demo_task
from ..tasks.intersect_set import intersect_set_task
from ..tasks.sort_list import sort_list_task
from ..tasks.sum_list import sum_list_task


class LanguageModelSimulationType(Enum):
    """
    Represents the type of language model simulation.
    """
    REALISTIC = 'realistic'
    """A realistic simulation of a language model."""

    DETERMINISTIC = 'deterministic'
    """A simplified simulation of a language model."""

    def get_factory_function(self, task: Task) -> Callable[[int], SimulatedLanguageModel]:
        """The factory function of a language model simulation type"""

        if task == sum_list_task:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_sum_list
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_sum_list
            raise

        if task == sort_list_task:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_sort_list
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_sort_list

        if task == intersect_set_task:
            if self == LanguageModelSimulationType.REALISTIC:
                return create_simulated_realistic_chat_gpt_intersect_set
            elif self == LanguageModelSimulationType.DETERMINISTIC:
                return create_simulated_deterministic_chat_gpt_intersect_set

        if task == count_demo_task:
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