from dataclasses import dataclass
from typing import Set

from .evaluator import Evaluator
from .operation import Operation
from .operation_type import OperationType


@dataclass(frozen=True)
class OperationRegistry:
    """
    Represents the registry for operations of a given problem.
    """

    operations: Set[Operation]
    """All operations that can be applied"""

    evaluator: Evaluator
    """The evaluator for ground truth evaluation"""

    @property
    def supported_types(self) -> Set[OperationType]:
        """Returns all supported types of operations by the registry's instance"""
        return {operation.type for operation in self.operations}
