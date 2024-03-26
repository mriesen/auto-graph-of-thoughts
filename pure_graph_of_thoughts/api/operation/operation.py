from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .operation_key import OperationKey
from ..language_model import Prompt
from ..state import State


@dataclass(frozen=True)
class Operation(OperationKey, ABC):
    """
    Represents an operation.
    """

    @property
    def key(self) -> OperationKey:
        """
        Returns the key of the operation.
        :return: operation key
        """
        return OperationKey(
                name=self.name,
                n_inputs=self.n_inputs,
                n_outputs=self.n_outputs,
                type=self.type
        )

    @property
    def is_scorable(self) -> bool:
        """
        Returns whether the operation can be scored.
        :return: operation is scorable
        """
        return False


@dataclass(frozen=True, eq=False, repr=False)
class ScoreOperation(Operation):
    """
    Represents a score operation.
    """
    pass


@dataclass(frozen=True, eq=False, repr=False)
class ScoreExecOperation(ScoreOperation):
    """
    Represents a score execution operation.
    The score is calculated by executing a defined score function.
    """

    score: Callable[[State, State], float]
    """The score function."""


@dataclass(frozen=True, eq=False, repr=False)
class ScorePromptOperation(ScoreOperation):
    """
    Represents a score prompt operation.
    The score is determined by prompting a language model.
    """

    prompt: Prompt
    """The score prompt"""

    transform_before: Callable[[State], State] = field(default=lambda state: state)
    """The transformation function applied on the input"""

    transform_after: Callable[[State], float] = field(default=lambda state: float(state['score']))
    """The transformation function applied on the output"""


@dataclass(frozen=True, eq=False, repr=False)
class PromptOperation(Operation):
    """
    Represents a prompt operation.
    A prompt operation is executed by prompting a language model.
    """

    prompt: Prompt
    """The prompt"""

    transform_before: Callable[[Sequence[State]], State] = field(
            default=lambda states: states[0] if states else {}
    )
    """The transformation function applied on the input"""

    transform_after: Callable[[State], Sequence[State]] = field(default=lambda state: [state])
    """The transformation function applied on the output"""

    score_operation: Optional[ScoreOperation] = field(default=None)
    """The score operation if the operation is scorable"""

    @property
    def is_scorable(self) -> bool:
        """
        Returns whether the operation can be scored.
        :return: scorable
        """
        return self.score_operation is not None


@dataclass(frozen=True, eq=False, repr=False)
class ExecOperation(Operation):
    """
    Represents an execution operation.
    The execution of such operations involve the invocation of a defined function.
    """
    execute: Callable[[Sequence[State]], Sequence[State]]
