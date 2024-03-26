from dataclasses import dataclass, field
from typing import Self, Optional

from .operation_node_schema import OperationNodeSchema
from ..node import Node
from ...internal.id import Id
from ...operation import Operation, OperationKey


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
    def of(cls, operation: Operation, id: Optional[Id] = None) -> Self:
        """
        Creates a node for a given operation.
        :param operation: operation of the node
        :param id: id of the node
        :return: new node
        """
        if id is not None:
            return cls(id=id, _operation=operation, _predecessors=[], _successors=[])
        else:
            return cls(_operation=operation, _predecessors=[], _successors=[])

    def append_operation(self, operation: Operation) -> Self:
        """
        Creates a node of a given operation and appends it as successor to the current one.
        :param operation: operation to append a new node for
        :return: new node
        """
        successor = self.of(operation)
        return self.append(successor)

    def to_schema(self) -> OperationNodeSchema:
        return OperationNodeSchema(
                id=self.id,
                operation_key=OperationKey(
                        name=self.operation.name,
                        n_inputs=self.operation.n_inputs,
                        n_outputs=self.operation.n_outputs,
                        type=self.operation.type
                )
        )

    def __hash__(self) -> int:
        return super().__hash__()
