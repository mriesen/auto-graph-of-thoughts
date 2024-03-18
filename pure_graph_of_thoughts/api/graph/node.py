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


@dataclass(frozen=True, eq=False)
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
    def is_source(self) -> bool:
        """
        Returns whether the node is the source of the graph.
        This is the case when a node has no predecessors.
        :return: node is source
        """
        return not self.predecessors

    @property
    def is_sink(self) -> bool:
        """
        Returns whether the node is a sink.
        This is the case when a node has no successors.
        :return: node is a sink
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

    def remove_successor(self, successor: Self) -> Self:
        """
        Removes a given successor from the current node.
        :param successor: successor to remove
        :return: current node
        """
        self._successors.remove(successor)
        return self

    def remove_all_successors(self) -> Self:
        """
        Removes all successors from the current node.
        :return: current node
        """
        self._successors.clear()
        return self

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.id))

    def __eq__(self, other: Any) -> bool:
        return other is not None and isinstance(other, Node) and other.__hash__() == self.__hash__()