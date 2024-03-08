from dataclasses import dataclass, field
from typing import Set, Sequence, Mapping, List

from .evaluator import Evaluator
from .operation import Operation
from .operation_type import OperationType


InvertedOperationIndex = Mapping[Operation, int]
"""Represents a mapping between operations and their indices."""


@dataclass(frozen=True)
class OperationRegistry:
    """
    Represents the registry for operations of a given problem.
    """

    operations: Sequence[Operation]
    """All operations that can be applied"""

    evaluator: Evaluator
    """The evaluator for ground truth evaluation"""

    def __post_init__(self) -> None:
        assert len(set(self.operations)) == len(self.operations), 'All operations must be unique'

    @property
    def supported_types(self) -> Set[OperationType]:
        """Returns all supported types of operations by the registry's instance"""
        return {operation.type for operation in self.operations}

    @property
    def inverted_operation_index(self) -> InvertedOperationIndex:
        return {
            operation: index for index, operation in enumerate(self.operations)
        }

    @property
    def n_operations(self) -> int:
        return len(self.operations)