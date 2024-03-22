from dataclasses import dataclass, field
from typing import Self

from ..node import Node
from ...operation import Operation


@dataclass(kw_only=True, eq=False)
class OperationNode(Node):
    """
    Represents a node in a graph of operations.
    """

    _operation: Operation
    """The node's operation"""

    _is_sealed: bool = field(default=False)
    """Whether the node is sealed"""

    @property
    def operation(self) -> Operation:
        """The operation of the node"""
        return self._operation

    @classmethod
    def of(cls, operation: Operation) -> Self:
        """
        Creates a node for a given operation.
        :param operation: operation of the node
        :return: new node
        """
        return cls(_operation=operation, _predecessors=[], _successors=[])

    def append_operation(self, operation: Operation) -> Self:
        """
        Creates a node of a given operation and appends it as successor to the current one.
        :param operation: operation to append a new node for
        :return: new node
        """
        successor = self.of(operation)
        return self.append(successor)

    def __hash__(self) -> int:
        return super().__hash__()
