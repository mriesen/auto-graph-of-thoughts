from math import floor
from typing import List, Sequence

from .baseline_result import BaselineResult
from .baseline_strategy import BaselineStrategy


class RandomBaselineStrategy(BaselineStrategy):
    """
    Random baseline strategy.
    Generates a random graph of operations and evaluates it with the given graph evaluator.
    """

    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResult:
        baseline_results: List[BaselineResult] = []
        for i in range(1, max_iterations + 1):
            current_baseline_result = self._generate_single(i)
            if stop_on_first_valid and current_baseline_result.is_valid:
                return current_baseline_result
            baseline_results.append(current_baseline_result)

        valid_baseline_results: Sequence[BaselineResult] = [
            baseline_result for baseline_result in baseline_results
            if baseline_result.is_valid
        ]
        return min(valid_baseline_results, key=lambda baseline_result: baseline_result.cost)

    def _generate_single(self, iteration: int) -> BaselineResult:
        graph_depth = self._random.randint(1, self._config.max_depth)
        max_breadth = self._config.max_breadth
        divergence_cutoff: int = floor(graph_depth * self._config.divergence_cutoff_factor)
        graph_of_operations = self._graph_generator.generate_random_graph(
                graph_depth, max_breadth, divergence_cutoff
        )
        self._graph_candidates.append(graph_of_operations)

        return self._graph_evaluator(graph_of_operations, iteration)
