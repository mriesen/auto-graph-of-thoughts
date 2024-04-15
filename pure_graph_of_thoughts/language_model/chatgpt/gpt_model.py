from enum import Enum


class GPTModel(Enum):
    """
    Represents a specific OpenAI GPT model.
    """

    GPT_35_TURBO_0125 = 'gpt-3.5-turbo-0125'
    GPT_35_TURBO_1106 = 'gpt-3.5-turbo-1106'
    GPT_35_TURBO_0613 = 'gpt-3.5-turbo-0613'
    GPT_35_TURBO_0301 = 'gpt-3.5-turbo-0301'

    @property
    def id(self) -> str:
        """The model ID"""
        return self.value


DEFAULT_GPT_MODEL: GPTModel = GPTModel.GPT_35_TURBO_0125
"""The default GPT model."""
