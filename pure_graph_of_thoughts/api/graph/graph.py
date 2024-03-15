from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Generic, Self, Set, Tuple, Sequence

from .node import Node

N = TypeVar('N', bound=Node)
"""The node type"""


@dataclass(frozen=True)
class Graph(ABC, Generic[N]):
    """
    Represents a graph.
    """

    source: N
    """The source node of the graph"""

    @classmethod
    def from_source(cls, source: N) -> Self:
        """
        Creates a new graph out of a given source node.
        :param source: source node of the graph
        :return: new graph
        """
        return cls(source)

    @property
    def nodes(self) -> Set[N]:
        """
        Returns all nodes in the graph.
        :return: all nodes
        """
        return set(self._get_nodes(self.source))

    @property
    def edges(self) -> Set[Tuple[N, N]]:
        """
        Returns all the edges in the graph.
        :return: all edges
        """
        return set(self._get_edges(self.source))

    @staticmethod
    def _get_nodes(current_node: N) -> Sequence[N]:
        return list(current_node.successors) + [
            node for successor in current_node.successors for node in
            Graph._get_nodes(successor)
        ]

    @staticmethod
    def _get_edges(current_node: N) -> Sequence[Tuple[N, N]]:
        return [(current_node, successor) for successor in current_node.successors] + [
            edge for successor in current_node.successors for edge in Graph._get_edges(successor)
        ]
