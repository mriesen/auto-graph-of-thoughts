import logging
from abc import ABC, abstractmethod
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
    _logger: logging.Logger

    def __init__(
            self,
            operations: Sequence[Operation],
            graph_evaluator: Callable[[GraphOfOperations, int], BaselineResult]
    ) -> None:
        """
        Instantiates a new baseline strategy.
        :param operations: operations
        :param graph_evaluator: graph evaluator to evaluate a generated graph of operations
        """

        self._operations = operations
        self._graph_generator = GraphGenerator(operations)
        self._graph_evaluator = graph_evaluator
        self._graph_candidates = []
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, max_iterations: int) -> BaselineResult:
        """
        Generates a baseline result by iteratively applying the strategy.
        :param max_iterations: maximum number of iterations
        :return: baseline result
        """

        result: BaselineResult = self._generate_single(0)
        for i in range(1, max_iterations):
            result = self._generate_single(i)
            if result.is_valid:
                return result
        return result

    @abstractmethod
    def _generate_single(self, iteration: int) -> BaselineResult:
        """
        Generates a single baseline result.
        :param iteration: current iteration
        :return: baseline result
        """
        pass
