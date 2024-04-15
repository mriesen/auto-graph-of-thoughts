import importlib.util
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Callable, cast, Union, Optional, Dict, Tuple

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from ..api.graph import Graph, Node, GraphSchema
from ..api.graph.operation import OperationNode, GraphOfOperations
from ..api.graph.thought import ThoughtNode, GraphOfThoughts
from ..api.internal.id import Id

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

    _node_label: Callable[[_N], _T] = field(default=lambda node: cast(_T, node))
    """Function to apply to a node to get the label of the node"""

    @staticmethod
    def _create_networkx_graph(
            graph: Graph[_N, _S],
            node_label: Callable[[_N], _T] = lambda node: cast(_T, node)
    ) -> Tuple[Dict[Id, _T], nx.DiGraph]:  # type: ignore
        nx_graph = nx.DiGraph()  # type: ignore
        nodes = graph.nodes
        edges = graph.edges
        node_ids = [node.id for node in nodes]
        edge_ids = [(edge[0].id, edge[1].id) for edge in edges]
        labels = {
            node.id: node_label(node) for node in nodes
        }

        nx_graph.add_nodes_from(node_ids)
        nx_graph.add_edges_from(edge_ids)
        return labels, nx_graph

    def plot_graph(self, graph: Graph[_N, _S], custom_node_label: Optional[Callable[[_N], _T]] = None) -> None:
        """
        Plots a given graph.
        :param graph: graph to plot
        :param custom_node_label: custom node label function to apply
        """

        node_label = self._node_label if custom_node_label is None else custom_node_label

        plt.tight_layout()
        labels, nx_graph = _GraphVisualizer._create_networkx_graph(graph, node_label)
        if _GraphVisualizer._is_pygraphviz_available():
            pos = graphviz_layout(nx_graph, prog='dot')
            nx.draw_networkx(nx_graph, pos=pos, arrows=True, labels=labels)
        else:
            nx.draw_networkx(nx_graph, arrows=True, labels=labels)
        plt.show()

    @staticmethod
    def _is_pygraphviz_available() -> bool:
        return importlib.util.find_spec('pygraphviz') is not None


_graph_of_operations_visualizer = _GraphVisualizer[str, OperationNode](
        lambda node: f'{node.operation.name}'
)

_graph_of_thoughts_visualizer = _GraphVisualizer[str, ThoughtNode](
        lambda node: f'{node.thought.state}'
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
