from abc import ABC, abstractmethod
from typing import Any

from .prompt import Prompt
from ..thought import State


class LanguageModel(ABC):
    """
    Represents a language model capable of answering a prompt.
    """

    @abstractmethod
    def prompt(self, prompt: Prompt, state: State) -> Any:
        """
        Processes a given prompt with a state.
        :param prompt: prompt to consume
        :param state: state to apply
        :return: answer of the language model
        """
        pass
