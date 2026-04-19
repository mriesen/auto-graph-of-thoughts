from math import floor
from typing import List, Sequence, Optional

from .baseline_strategy import BaselineStrategy
from .model import BaselineResultSummary, BaselineIterationResult


class RandomBaselineStrategy(BaselineStrategy):
    """
    Random baseline strategy.
    Generates a random graph of operations and evaluates it with the given graph evaluator.
    """

    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResultSummary:
        iteration_results: List[BaselineIterationResult] = []
        for i in range(1, max_iterations + 1):
            current_iteration_result = self._generate_single(i)
            iteration_results.append(current_iteration_result)
            if stop_on_first_valid and current_iteration_result.is_valid:
                return BaselineResultSummary(
                        results=iteration_results,
                        final_result_index=iteration_results.index(current_iteration_result),
                        max_iterations=max_iterations,
                        stop_on_first_valid=True
                )

        final_result: Optional[BaselineIterationResult] = self._find_valid_min_cost(iteration_results)
        return BaselineResultSummary(
                results=iteration_results,
                final_result_index=iteration_results.index(
                        final_result
                ) if final_result is not None else None,
                max_iterations=max_iterations
        )

    def _generate_single(self, iteration: int) -> BaselineIterationResult:
        graph_depth = self._random.randint(1, self._config.max_depth)
        max_breadth = self._config.max_breadth
        divergence_cutoff: int = floor(graph_depth * self._config.divergence_cutoff_factor)
        graph_of_operations = self._graph_generator.generate_random_graph(
                graph_depth, max_breadth, divergence_cutoff
        )
        self._graph_candidates.append(graph_of_operations)

        return self._evaluate_graph(graph_of_operations, iteration)
