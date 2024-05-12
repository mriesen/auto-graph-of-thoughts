from enum import Enum
from typing import Callable

from ..language_model import create_simulated_realistic_chat_gpt_sum_list, create_simulated_deterministic_chat_gpt_sum_list
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel


class LanguageModelSimulationType(Enum):
    """
    Represents the type of language model simulation.
    """
    REALISTIC = 'realistic'
    """A realistic simulation of a language model."""

    DETERMINISTIC = 'deterministic'
    """A simplified simulation of a language model."""

    @property
    def factory_function(self) -> Callable[[int], SimulatedLanguageModel]:
        """The factory function of a language model simulation type"""
        if self == LanguageModelSimulationType.REALISTIC:
            return create_simulated_realistic_chat_gpt_sum_list
        elif self == LanguageModelSimulationType.DETERMINISTIC:
            return create_simulated_deterministic_chat_gpt_sum_list
        raise


class LanguageModelSimulationTypeException(Exception):
    """
    An exception that is thrown in context of a language model simulation type.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)