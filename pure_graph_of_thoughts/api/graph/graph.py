from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Generic, Self, Set, Tuple, Sequence, Dict, List

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

    @property
    def sinks(self) -> Sequence[N]:
        """
        Returns all the sinks of the graph.
        :return: all sinks
        """
        return [node for node in self.nodes if node.is_sink]

    @property
    def layers(self) -> Sequence[Sequence[N]]:
        """
        Returns all the layers of nodes in the graph.
        :return: all layers of nodes
        """
        layers: Dict[int, List[N]] = defaultdict(list)
        for node in self.nodes:
            layers[node.depth].append(node)
        layer_matrix = [
            layer for depth, layer in sorted(layers.items())
        ]
        return layer_matrix

    @property
    def depth(self) -> int:
        """
        Returns the depth of the graph.
        :return: the depth of the graph
        """
        return max([node.depth for node in self.nodes], default=0)

    @staticmethod
    def _get_nodes(current_node: N) -> Sequence[N]:
        return [current_node] + [
            node
            for successor in current_node.successors
            for node in Graph._get_nodes(successor)
        ]

    @staticmethod
    def _get_edges(current_node: N) -> Sequence[Tuple[N, N]]:
        return [(current_node, successor) for successor in current_node.successors] + [
            edge for successor in current_node.successors for edge in Graph._get_edges(successor)
        ]
