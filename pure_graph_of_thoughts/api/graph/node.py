import itertools
from dataclasses import dataclass, field
from typing import List, Iterator, Self

from ..operation import Operation

_node_id_generator: Iterator[int] = itertools.count(0)


def _next_node_id() -> int:
    """
    Generates the next node ID.
    :return: next node ID
    """
    return next(_node_id_generator)


@dataclass(frozen=True, eq=False)
class Node:
    """
    Represents a node in a graph of operations.
    """

    operation: Operation
    """The node's operation"""

    predecessors: List[Self]
    """The predecessors of the node"""

    successors: List[Self]
    """The successors of the node"""

    id: int = field(default_factory=_next_node_id)
    """The ID of the node for unique identification"""

    @property
    def depth(self) -> int:
        """
        Returns the depth at which the node is.
        :return: depth
        """
        if self.predecessors is None or len(self.predecessors) == 0:
            return 0
        return self.predecessors[0].depth + 1

    @property
    def is_root(self) -> bool:
        """
        Returns whether the node is the root of the graph.
        This is the case when a node has no predecessors.
        :return: node is root
        """
        return not self.predecessors

    @property
    def is_leaf(self) -> bool:
        """
        Returns whether the node is a leaf.
        This is the case when a node has no successors.
        :return: node is a leaf
        """
        return not self.successors

    @classmethod
    def of_operation(cls, operation: Operation) -> Self:
        """
        Creates a node for a given operation.
        :param operation: operation of the node
        :return: new node
        """
        return cls(operation=operation, predecessors=[], successors=[])

    def append_operation(self, operation: Operation) -> Self:
        """
        Creates a node of a given operation and appends it as successor to the current one.
        :param operation: operation to append a new node for
        :return: new node
        """
        successor = self.of_operation(operation)
        return self.append(successor)

    def append(self: Self, successor: Self) -> Self:
        """
        Appends a successor to the current node.
        :param successor: successor node to append
        :return: successor node
        """
        successor.predecessors.append(self)
        self.successors.append(successor)
        return successor

    def __hash__(self) -> int:
        return self.id