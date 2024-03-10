import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Self, Sequence, Callable, Any


def node_id_generator() -> Callable[[], int]:
    """
    Creates a node ID generator.
    :return: function that returns the next node ID
    """
    id_iterator = itertools.count(0)
    return lambda: next(id_iterator)


@dataclass(frozen=True)
class Node(ABC):
    """
    Abstract representation of a node in a graph.
    """

    _predecessors: List[Self]
    """The predecessors of the node"""

    _successors: List[Self]
    """The successors of the node"""

    @property
    @abstractmethod
    def id(self) -> int:
        """The ID of the node"""
        pass

    @property
    def predecessors(self) -> Sequence[Self]:
        return self._predecessors

    @property
    def successors(self) -> Sequence[Self]:
        return self._successors

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

    def append(self: Self, successor: Self) -> Self:
        """
        Appends a successor to the current node.
        :param successor: successor node to append
        :return: successor node
        """
        successor._predecessors.append(self)
        self._successors.append(successor)
        return successor

    def append_all(self, successors: Sequence[Self]) -> Sequence[Self]:
        """
        Appends multiple successors to the current node.
        :param successors: successor nodes to append
        :return: successor nodes
        """
        return [self.append(successor) for successor in successors]

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.id))

    def __eq__(self, other: Any) -> bool:
        return other is not None and isinstance(other, Node) and other.__hash__() == self.__hash__()