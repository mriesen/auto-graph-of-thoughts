from dataclasses import dataclass, field
from typing import Self

from ..node import Node, node_id_generator
from ...operation import Operation

_next_operation_node_id = node_id_generator()


@dataclass(frozen=True)
class OperationNode(Node):
    """
    Represents a node in a graph of operations.
    """

    operation: Operation
    """The node's operation"""

    _id: int = field(default_factory=_next_operation_node_id)
    """The ID of the node for unique identification"""

    @property
    def id(self) -> int:
        """The ID of the node"""
        return self._id

    @classmethod
    def of(cls, operation: Operation) -> Self:
        """
        Creates a node for a given operation.
        :param operation: operation of the node
        :return: new node
        """
        return cls(operation=operation, _predecessors=[], _successors=[])

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
