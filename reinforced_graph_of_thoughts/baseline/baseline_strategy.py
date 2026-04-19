import logging
from abc import ABC, abstractmethod
from random import Random
from typing import Sequence, Callable, List, Optional

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.operation import Operation
from .baseline_config import BaselineConfig
from .graph_generator import GraphGenerator
from .model import BaselineIterationResult, BaselineResultSummary


class BaselineStrategy(ABC):
    """
    Represents a baseline strategy to generate a static graph of operations.
    """

    _config: BaselineConfig
    _operations: Sequence[Operation]
    _graph_generator: GraphGenerator
    _evaluate_graph: Callable[[GraphOfOperations, int], BaselineIterationResult]
    _graph_candidates: List[GraphOfOperations]
    _random: Random
    _logger: logging.Logger

    def __init__(self, config: BaselineConfig) -> None:
        """
        Instantiates a new baseline strategy.
        :param config: baseline strategy configuration
        """
        self._config = config
        self._operations = config.operations
        self._graph_generator = GraphGenerator(config.operations, config.seed)
        self._evaluate_graph = config.evaluate_graph
        self._graph_candidates = []
        self._random = Random(config.seed)
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResultSummary:
        """
        Generates a baseline result by iteratively applying the strategy.
        :param max_iterations: maximum number of iterations
        :param stop_on_first_valid: whether to stop on the first valid result (default: False)
        :return: baseline result
        """
        pass

    @abstractmethod
    def _generate_single(self, iteration: int) -> BaselineIterationResult:
        """
        Generates a single baseline result.
        :param iteration: current iteration
        :return: baseline result
        """
        pass

    @staticmethod
    def _find_valid_min_cost(iteration_results: Sequence[BaselineIterationResult]) -> Optional[BaselineIterationResult]:
        """
        Finds the valid iteration result with the lowest cost.
        :param iteration_results: iteration results
        :return: valid iteration result with the lowest cost (if present)
        """
        valid_results: Sequence[BaselineIterationResult] = [
            iteration_result for iteration_result in iteration_results
            if iteration_result.is_valid
        ]
        return min(
                valid_results,
                key=lambda baseline_result: baseline_result.cost
        ) if len(valid_results) > 0 else None
