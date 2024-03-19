from dataclasses import dataclass

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations


@dataclass(frozen=True)
class BaselineResult:
    """
    Represents a result of a baseline strategy run.
    """

    graph_of_operations: GraphOfOperations
    """The generated graph of operations"""

    is_valid: bool
    """Whether the result is valid"""

    cost: float
    """The cost of the graph of operations execution"""

    iteration: int
    """The iteration of the result"""
