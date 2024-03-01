from dataclasses import dataclass

from .thought_node import ThoughtNode
from ..graph import Graph
from ..operation import OperationNode


@dataclass(frozen=True)
class GraphOfThoughts(Graph[ThoughtNode]):
    pass
