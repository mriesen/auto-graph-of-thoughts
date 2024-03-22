from typing import Dict, Callable

from ..api.language_model import LanguageModel, Prompt, LanguageModelException
from ..api.state import State


class MockLanguageModel(LanguageModel):
    """
    A mock for the language model.
    The behavior by prompt can be set on instantiation.
    """

    _mocked_behaviors: Dict[Prompt, Callable[[Prompt, State], State]]

    def __init__(self, mocked_behaviors: Dict[Prompt, Callable[[Prompt, State], State]]):
        """
        Instantiates a new mock language model.
        :param mocked_behaviors: mocked behaviors by prompt
        """
        self._mocked_behaviors = mocked_behaviors

    def prompt(self, prompt: Prompt, state: State) -> State:
        if prompt not in self._mocked_behaviors:
            raise LanguageModelException(f'No mocked behavior found for prompt {prompt}')
        mocked_behavior = self._mocked_behaviors[prompt]
        return mocked_behavior(prompt, state)
