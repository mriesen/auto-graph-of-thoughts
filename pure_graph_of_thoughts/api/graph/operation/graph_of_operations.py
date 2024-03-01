from dataclasses import dataclass

from ..graph import Graph
from .operation_node import OperationNode


@dataclass(frozen=True)
class GraphOfOperations(Graph[OperationNode]):
    pass
