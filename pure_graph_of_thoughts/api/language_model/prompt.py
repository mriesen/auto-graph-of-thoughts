import json
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Sequence

from .example import Example
from ..state import State


@dataclass(frozen=True)
class Prompt:
    """
    Represents a prompt for a language model.
    A prompt consists of an instruction
    and optional examples to provide to the language model for in-context learning.
    """

    instruction: str
    """The primary instruction"""

    examples: Sequence[Example] = field(default_factory=lambda: [])
    """The examples to provide to the language model for in-context learning"""

    def __post_init__(self) -> None:
        if 'JSON' not in self.instruction:
            raise PromptException('Instruction must contain the word "JSON"')

    def __str__(self) -> str:
        return dedent(
                '''
            <Instruction>{instruction}</Instruction>
            <Examples>{examples}</Examples>
            '''.format(
                        instruction=self.instruction,
                        examples='\n'.join(map(lambda example: str(example), self.examples))
                )
        )

    def for_input(self, input_state: State) -> str:
        """
        Returns the built prompt for a given input state.
        :param input_state: input state
        :return: prompt as string
        """
        return '{prompt}\nInput: {state}'.format(prompt=self.__str__(), state=json.dumps(input_state))

    def __hash__(self) -> int:
        return hash((self.__str__()))


class PromptException(Exception):
    """
    An exception raised due to a prompt.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
