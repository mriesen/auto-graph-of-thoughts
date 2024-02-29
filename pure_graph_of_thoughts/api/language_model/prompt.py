import json
from dataclasses import dataclass, field
from textwrap import dedent
from typing import List

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

    examples: List[Example] = field(default_factory=lambda: [])
    """The examples to provide to the language model for in-context learning"""

    def __post_init__(self) -> None:
        assert 'JSON' in self.instruction, 'Instruction must contain the word "JSON"'

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
