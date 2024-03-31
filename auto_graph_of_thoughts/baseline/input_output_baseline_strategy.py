from typing import List, Optional, Callable

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.operation import Operation
from . import BaselineConfig
from .baseline_strategy import BaselineStrategy
from .model import BaselineResultSummary, BaselineIterationResult


class InputOutputBaselineStrategy(BaselineStrategy):
    """
    Input-Output (IO) baseline strategy.
    Generates a graph of operations containing a single operation node.
    """

    _operation: Operation

    def __init__(
            self,
            evaluate_graph: Callable[[GraphOfOperations, int], BaselineIterationResult],
            operation: Operation
    ) -> None:
        """
        Instantiates a new Input-Output baseline strategy.
        :param evaluate_graph: graph evaluation function
        :param operation: operation to use
        """
        super().__init__(BaselineConfig(
                max_depth=1,
                max_breadth=1,
                divergence_cutoff_factor=0,
                operations=[operation],
                evaluate_graph=evaluate_graph
        ))
        self._operation = operation

    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResultSummary:
        iteration_results: List[BaselineIterationResult] = []
        for i in range(1, max_iterations + 1):
            iteration_result = self._generate_single(i)
            iteration_results.append(iteration_result)
            if stop_on_first_valid and iteration_result.is_valid:
                return BaselineResultSummary(
                        results=iteration_results,
                        final_result_index=iteration_results.index(iteration_result),
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
        graph_of_operations: GraphOfOperations = self._graph_generator.generate_singleton_graph(self._operation)
        return self._evaluate_graph(graph_of_operations, iteration)
