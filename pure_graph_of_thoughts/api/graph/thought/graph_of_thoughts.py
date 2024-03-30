from dataclasses import dataclass
from typing import Self, Dict, Any

from .thought_graph_schema import ThoughtGraphSchema
from .thought_node import ThoughtNode
from .thought_node_schema import ThoughtNodeSchema
from ..graph import Graph
from ..graph_schema import GraphSchema
from ...state import State
from ...thought import Thought


@dataclass
class GraphOfThoughts(Graph[ThoughtNode, ThoughtGraphSchema]):
    """
    Represents a graph of thoughts.
    """

    @classmethod
    def from_init_state(cls, init_state: State) -> Self:
        """
        Creates a graph of thoughts from an initial state.
        :param init_state: initial state
        :return: created graph of thoughts
        """
        source = ThoughtNode.of(Thought(state=init_state))
        return cls.from_source(source)

    @classmethod
    def from_schema(cls, schema: GraphSchema[ThoughtNodeSchema]) -> Self:
        """
        Constructs a graph of thoughts from a given graph schema and a sequence of operation nodes.
        :param schema: graph schema
        :return: constructed graph of thoughts
        """
        nodes = [
            ThoughtNode.of(
                    id=node.id,
                    thought=Thought(
                            state=node.state,
                            score=node.score,
                            origin_id=node.origin_id,
                    )
            ) for node in schema.nodes
        ]
        return cls._construct_graph(nodes, schema.edges)

    def to_schema(self) -> ThoughtGraphSchema:
        return super()._to_schema(ThoughtGraphSchema)

    def __deepcopy__(self, memo: Dict[Any, Any]) -> Self:
        return self.from_schema(self.to_schema())
