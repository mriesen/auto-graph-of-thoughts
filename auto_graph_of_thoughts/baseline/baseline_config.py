from dataclasses import dataclass, field
from typing import Sequence, Callable, Optional

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.operation import Operation
from .model import BaselineIterationResult


@dataclass(frozen=True, kw_only=True)
class BaselineConfig:
    """
    The configuration of a baseline strategy execution.
    """

    max_breadth: int
    """The maximum breadth of the graph"""

    max_depth: int
    """The maximum depth of the graph"""

    divergence_cutoff_factor: float
    """The factor used to calculate the divergence cutoff"""

    operations: Sequence[Operation]
    """The list of available operations"""

    evaluate_graph: Callable[[GraphOfOperations, int], BaselineIterationResult]
    """The graph evaluator to evaluate a generated graph of operations"""

    seed: Optional[int] = field(default=None)
    """The seed for random number generator"""
