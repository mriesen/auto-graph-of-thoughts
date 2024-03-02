import importlib.util
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Callable, cast, Union

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from ..api.graph import Graph, Node
from ..api.graph.operation import OperationNode, GraphOfOperations
from ..api.graph.thought import ThoughtNode, GraphOfThoughts

N = TypeVar('N', bound=Node)
T = TypeVar('T', bound=Any)


@dataclass(frozen=True)
class GraphVisualizer(Generic[T, N]):
    _transform_node: Callable[[N], T] = field(default=lambda node: cast(T, node))
    """Transform function to apply to a node before constructing a networkx graph"""

    @staticmethod
    def _create_networkx_graph(
            graph: Graph[N],
            transform_node: Callable[[N], T] = lambda node: cast(T, node)
    ) -> nx.DiGraph:  # type: ignore
        nx_graph = nx.DiGraph()  # type: ignore
        nodes = list(map(transform_node, graph.nodes))
        edges = [(transform_node(edge[0]), transform_node(edge[1])) for edge in graph.edges]

        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)
        return nx_graph

    def plot_graph(self, graph: Graph[N]) -> None:
        plt.tight_layout()
        nx_graph = GraphVisualizer._create_networkx_graph(graph, self._transform_node)
        if GraphVisualizer._is_pygraphviz_available():
            pos = graphviz_layout(nx_graph, prog='dot')
            nx.draw_networkx(nx_graph, pos=pos, arrows=True)
        else:
            nx.draw_networkx(nx_graph, arrows=True)
        plt.show()

    @staticmethod
    def _is_pygraphviz_available() -> bool:
        return importlib.util.find_spec('pygraphviz') is not None


graph_of_operations_visualizer = GraphVisualizer[str, OperationNode](
        lambda node: f'{node.id}-{node.operation.name}'
)

graph_of_thoughts_visualizer = GraphVisualizer[str, ThoughtNode](
        lambda node: f'{node.id}-{node.thought.state}'
)


def plot_graph(graph: Union[GraphOfOperations, GraphOfThoughts]) -> None:
    if isinstance(graph, GraphOfOperations):
        graph_of_operations_visualizer.plot_graph(graph)
    if isinstance(graph, GraphOfThoughts):
        graph_of_thoughts_visualizer.plot_graph(graph)
