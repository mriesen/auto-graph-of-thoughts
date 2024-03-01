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

    root: N
    """The root node of the graph"""

    @classmethod
    def from_root(cls, root: N) -> Self:
        """
        Creates a new graph out of a given root node.
        :param root: root node of the graph
        :return: new graph
        """
        return cls(root)

    @property
    def nodes(self) -> Set[N]:
        """
        Returns all nodes in the graph.
        :return: all nodes
        """
        return set(self._get_nodes(self.root))

    @property
    def edges(self) -> Set[Tuple[N, N]]:
        """
        Returns all the edges in the graph.
        :return: all edges
        """
        return set(self._get_edges(self.root))

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
