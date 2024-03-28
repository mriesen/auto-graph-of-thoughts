import logging
from abc import ABC, abstractmethod
from random import Random
from typing import Sequence, Callable, List, Optional

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.operation import Operation
from .baseline_result import BaselineResult
from .graph_generator import GraphGenerator


class BaselineStrategy(ABC):
    """
    Represents a baseline strategy to generate a static graph of operations.
    """

    _MAX_DEPTH: int = 8
    _MAX_BREADTH: int = 8
    _DIVERGENCE_CUTOFF_FACTOR: float = 0.4

    _operations: Sequence[Operation]
    _graph_generator: GraphGenerator
    _graph_evaluator: Callable[[GraphOfOperations, int], BaselineResult]
    _graph_candidates: List[GraphOfOperations]
    _random: Random
    _logger: logging.Logger

    def __init__(
            self,
            operations: Sequence[Operation],
            graph_evaluator: Callable[[GraphOfOperations, int], BaselineResult],
            seed: Optional[int] = None
    ) -> None:
        """
        Instantiates a new baseline strategy.
        :param operations: operations
        :param graph_evaluator: graph evaluator to evaluate a generated graph of operations
        :param seed: seed for random number generator
        """
        self._operations = operations
        self._graph_generator = GraphGenerator(operations, seed)
        self._graph_evaluator = graph_evaluator
        self._graph_candidates = []
        self._random = Random(seed)
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResult:
        """
        Generates a baseline result by iteratively applying the strategy.
        :param max_iterations: maximum number of iterations
        :param stop_on_first_valid: whether to stop on the first valid result (default: False)
        :return: baseline result
        """
        pass

    @abstractmethod
    def _generate_single(self, iteration: int) -> BaselineResult:
        """
        Generates a single baseline result.
        :param iteration: current iteration
        :return: baseline result
        """
        pass
