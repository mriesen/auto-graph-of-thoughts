import json
from dataclasses import dataclass

from ..state import State


@dataclass(frozen=True)
class Example:
    """
    Represents an example for a prompt's input and output.
    """

    input: State
    """The input state"""

    output: State
    """The output state"""

    def __str__(self) -> str:
        return '''
            <Example>
                Input: {input}
                Output: {output}
            </Example>
            '''.format(
                input=json.dumps(self.input),
                output=json.dumps(self.output)
        )
