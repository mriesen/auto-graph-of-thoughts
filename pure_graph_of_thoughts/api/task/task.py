from dataclasses import dataclass
from typing import Set, Sequence, Mapping

from .evaluator import Evaluator
from ..operation import OperationKey, Operation, OperationType

InvertedOperationIndex = Mapping[OperationKey, int]
"""Represents a mapping between operations and their indices."""


@dataclass(frozen=True)
class Task:
    """
    Represents a task.
    A task consists of a list of operations and an evaluator.
    """

    operations: Sequence[Operation]
    """All operations that can be applied"""

    evaluator: Evaluator
    """The evaluator for ground truth evaluation"""

    def __post_init__(self) -> None:
        if len(set(self.operations)) != len(self.operations):
            raise TaskException('All operations must be unique')

    @property
    def supported_types(self) -> Set[OperationType]:
        """Returns all supported types of operations of the task"""
        return {operation.type for operation in self.operations}

    @property
    def inverted_operation_index(self) -> InvertedOperationIndex:
        return {
            operation.key: index for index, operation in enumerate(self.operations)
        }

    @property
    def n_operations(self) -> int:
        return len(self.operations)


class TaskException(Exception):
    """
    An exception raised when a task instance cannot be constructed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
