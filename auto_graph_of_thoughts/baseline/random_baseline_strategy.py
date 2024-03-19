from math import floor
from random import randint

from .baseline_result import BaselineResult
from .baseline_strategy import BaselineStrategy


class RandomBaselineStrategy(BaselineStrategy):
    """
    Random baseline strategy.
    Generates a random graph of operations and evaluates it with the given graph evaluator.
    """

    def _generate_single(self, iteration: int) -> BaselineResult:
        graph_depth = randint(1, self._MAX_DEPTH)
        max_breadth = self._MAX_BREADTH
        divergence_cutoff: int = floor(graph_depth * self._DIVERGENCE_CUTOFF_FACTOR)
        graph_of_operations = self._graph_generator.generate_random_graph(graph_depth, max_breadth, divergence_cutoff)
        self._graph_candidates.append(graph_of_operations)

        baseline_result = self._graph_evaluator(graph_of_operations, iteration)
        return baseline_result
