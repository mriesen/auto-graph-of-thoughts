import importlib.util
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Callable, cast, Union, Optional

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from ..api.graph import Graph, Node, GraphSchema
from ..api.graph.operation import OperationNode, GraphOfOperations
from ..api.graph.thought import ThoughtNode, GraphOfThoughts

_N = TypeVar('_N', bound=Node)
"""The type of node"""

_S = TypeVar('_S', bound=GraphSchema[Any])
"""The graph schema type"""

_T = TypeVar('_T', bound=Any)
"""The type of the visual representation of the node"""


@dataclass(frozen=True)
class _GraphVisualizer(Generic[_T, _N]):
    """
    Visualizer for graphs.
    """

    _transform_node: Callable[[_N], _T] = field(default=lambda node: cast(_T, node))
    """Transform function to apply to a node before constructing a networkx graph"""

    @staticmethod
    def _create_networkx_graph(
            graph: Graph[_N, _S],
            transform_node: Callable[[_N], _T] = lambda node: cast(_T, node)
    ) -> nx.DiGraph:  # type: ignore
        nx_graph = nx.DiGraph()  # type: ignore
        nodes = list(map(transform_node, graph.nodes))
        edges = [(transform_node(edge[0]), transform_node(edge[1])) for edge in graph.edges]

        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)
        return nx_graph

    def plot_graph(self, graph: Graph[_N, _S], custom_transform_node: Optional[Callable[[_N], _T]] = None) -> None:
        """
        Plots a given graph.
        :param graph: graph to plot
        :param custom_transform_node: custom transform node function to apply
        """

        transform_node = self._transform_node if custom_transform_node is None else custom_transform_node

        plt.tight_layout()
        nx_graph = _GraphVisualizer._create_networkx_graph(graph, transform_node)
        if _GraphVisualizer._is_pygraphviz_available():
            pos = graphviz_layout(nx_graph, prog='dot')
            nx.draw_networkx(nx_graph, pos=pos, arrows=True)
        else:
            nx.draw_networkx(nx_graph, arrows=True)
        plt.show()

    @staticmethod
    def _is_pygraphviz_available() -> bool:
        return importlib.util.find_spec('pygraphviz') is not None


_graph_of_operations_visualizer = _GraphVisualizer[str, OperationNode](
        lambda node: f'{node.id}-{node.operation.name}'
)

_graph_of_thoughts_visualizer = _GraphVisualizer[str, ThoughtNode](
        lambda node: f'{node.id}-{node.thought.state}'
)


def plot_graph(
        graph: Union[GraphOfOperations, GraphOfThoughts],
        transform_node: Optional[Callable[[Union[OperationNode, ThoughtNode]], Any]] = None
) -> None:
    """
    Plots a graph.

    :param graph: graph to plot
    :param transform_node: transform node function to apply to each node
    """

    if isinstance(graph, GraphOfOperations):
        _graph_of_operations_visualizer.plot_graph(graph, transform_node)
    if isinstance(graph, GraphOfThoughts):
        _graph_of_thoughts_visualizer.plot_graph(graph, transform_node)
