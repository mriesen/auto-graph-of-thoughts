from abc import ABC
from dataclasses import dataclass
from typing import List, Self, Sequence, Any

from ..internal.id import Identifiable
from ..internal.seal import Sealable, mutating, MutationScope


@dataclass(kw_only=True, eq=False)
class Node(Identifiable, Sealable, ABC):
    """
    Abstract representation of a node in a graph.
    """

    _predecessors: List[Self]
    """The predecessors of the node"""

    _successors: List[Self]
    """The successors of the node"""

    @property
    def predecessors(self) -> Sequence[Self]:
        """The predecessors of the node"""
        return self._predecessors

    @property
    def successors(self) -> Sequence[Self]:
        """The successors of the node"""
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

    @mutating(scope=MutationScope.ALL)
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

    def seal(self) -> None:
        if not self.is_sealed:
            super().seal()
        for unsealed_successor in [successor for successor in self._successors if not successor.is_sealed]:
            unsealed_successor.seal()
        for unsealed_predecessor in [predecessor for predecessor in self._predecessors if not predecessor.is_sealed]:
            unsealed_predecessor.seal()

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.id))

    def __eq__(self, other: Any) -> bool:
        return other is not None and isinstance(other, Node) and other.__hash__() == self.__hash__()
