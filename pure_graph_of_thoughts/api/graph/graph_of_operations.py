from dataclasses import dataclass
from typing import List, Set, Tuple, Self

from .node import Node


@dataclass(frozen=True)
class GraphOfOperations:
    """
    Represents a graph of operations.
    """

    root: Node
    """The root node of the graph"""

    @classmethod
    def from_root(cls, root: Node) -> Self:
        """
        Creates a new graph out of a given root node.
        :param root: root node of the graph
        :return: new graph
        """
        return cls(root)

    @property
    def nodes(self) -> Set[Node]:
        """
        Returns all nodes in the graph.
        :return: all nodes
        """
        return set(self._get_nodes(self.root))

    @property
    def edges(self) -> Set[Tuple[Node, Node]]:
        """
        Returns all the edges in the graph.
        :return: all edges
        """
        return set(self._get_edges(self.root))

    @staticmethod
    def _get_nodes(current_node: Node) -> List[Node]:
        return current_node.successors + [node for child in current_node.successors for node in
                                          GraphOfOperations._get_nodes(child)]

    @staticmethod
    def _get_edges(current_node: Node) -> List[Tuple[Node, Node]]:
        return (
                [(current_node, child) for child in current_node.successors]
                + [edge for child in current_node.successors for edge in GraphOfOperations._get_edges(child)]
        )
