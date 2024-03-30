import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Generic, Self, Set, Tuple, Sequence, Dict, List, Mapping, Type, Any

from .graph_schema import GraphSchema
from .node import Node
from .node_schema import NodeSchema
from ..internal.id import Id
from ..internal.seal import Sealable
from ..operation import Operation

N = TypeVar('N', bound=Node)
"""The node type"""

S = TypeVar('S', bound=GraphSchema[Any])
"""The graph schema type"""


@dataclass(kw_only=True)
class Graph(Sealable, ABC, Generic[N, S]):
    """
    Represents a graph.
    """

    _source: N
    """The source node of the graph"""

    @property
    def source(self) -> N:
        """The source node of the graph"""
        return self._source

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
        return [
            layer for depth, layer in sorted(layers.items())
        ]

    @property
    def depth(self) -> int:
        """
        Returns the depth of the graph.
        :return: the depth of the graph
        """
        return max([node.depth for node in self.nodes], default=0)

    def seal(self) -> None:
        super().seal()
        self._source.seal()

    @abstractmethod
    def to_schema(self) -> S:
        pass

    def _to_schema(self, schema_cls: Type[S]) -> S:
        """
        Converts the graph into its schematic form.
        :return: graph schema
        """

        edges: Sequence[Tuple[Id, Id]] = [
            (a.id, b.id) for (a, b) in self.edges
        ]
        nodes: Sequence[NodeSchema] = [
            node.to_schema() for node in self.nodes
        ]
        return schema_cls(edges=edges, nodes=nodes)

    @classmethod
    def from_source(cls, source: N) -> Self:
        """
        Creates a new graph out of a given source node.
        :param source: source node of the graph
        :return: new graph
        """
        return cls(_source=source)

    @classmethod
    def _construct_graph(cls, nodes: Sequence[N], edges: Sequence[Tuple[Id, Id]]) -> Self:
        """
        Constructs a graph out of a given sequence of nodes and edges.
        :param nodes: nodes of the graph
        :param edges: edges of the graph
        :param operations: available operations
        :return: constructed graph
        """
        nodes_by_id: Mapping[Id, N] = {
            node.id: node for node in nodes
        }
        for (id_current, id_successor) in edges:
            current_node = nodes_by_id[id_current]
            successor_node = nodes_by_id[id_successor]
            current_node.append(successor_node)

        source_node = [node for node in nodes if node.is_source][0]
        return cls.from_source(source_node)

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

    def clone(self) -> Self:
        """
        Clones the current graph and returns a deep copy of the graph.
        :return: deep copy
        """
        return copy.deepcopy(self)
