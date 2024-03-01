from abc import ABC, abstractmethod

from .prompt import Prompt
from ..state import State


class LanguageModel(ABC):
    """
    Represents a language model capable of answering a prompt.
    """

    @abstractmethod
    def prompt(self, prompt: Prompt, state: State) -> State:
        """
        Processes a given prompt with a state.
        :param prompt: prompt to consume
        :param state: state to apply
        :return: answer of the language model
        """
        pass
