from math import exp, floor, ceil
from random import random, randint
from typing import Sequence, Callable, Optional

from .baseline_result import BaselineResult
from .baseline_strategy import BaselineStrategy
from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.operation import Operation


class SimulatedAnnealingBaselineStrategy(BaselineStrategy):
    """
    Simulated annealing baseline strategy.
    A decreasing temperature parameter controls the simulated annealing behavior.
    The first graph is generated randomly.
    After the first generation, a neighbor of the graph is generated.
    A neighbor is defined as a serial composition of a subgraph of the original graph and a randomly generated subgraph.
    The current graph is clipped at a random depth and extended with a subgraph of a random depth.
    If the neighbor could not be generated, the generation is re-tried until a certain threshold
    is reached and a complete re-generation is applied.
    """

    _GRAPH_REGENERATION_THRESHOLD = 10

    _temperature: float
    _cooling_factor: float
    _best_energy: float
    _best_graph: Optional[GraphOfOperations]
    _current_graph: Optional[GraphOfOperations]

    def __init__(self,
                 operations: Sequence[Operation],
                 cooling_factor: float,
                 graph_evaluator: Callable[[GraphOfOperations, int], BaselineResult]) -> None:
        """
        Instantiates a simulated annealing baseline strategy.
        :param operations: operations
        :param cooling_factor: cooling factor to decrease the temperature with
        :param graph_evaluator: evaluator for the generated graph of operations
        """

        super().__init__(operations, graph_evaluator)

        self._temperature = 1.0
        self._cooling_factor = cooling_factor
        self._best_energy = 0
        self._current_graph = None

    def _generate_single(self, iteration: int) -> BaselineResult:

        self._temperature *= self._cooling_factor
        new_graph = self._create_neighbor(self._current_graph)
        baseline_result = self._graph_evaluator(new_graph, iteration)
        energy = float(baseline_result.is_valid) / baseline_result.cost
        probability: float = exp(-energy / self._temperature)

        if energy > self._best_energy or random() <= probability:
            self._current_graph = new_graph

        return baseline_result

    def _create_neighbor(self, graph: Optional[GraphOfOperations]) -> GraphOfOperations:
        if graph is None:
            return self._create_complete_graph()

        operation_matrix = graph.operation_matrix

        prev_depth = len(operation_matrix)

        clip_depth = randint(1, prev_depth)
        clip_layers = operation_matrix[:clip_depth - 1]

        if len(clip_layers) == 0:
            return self._create_neighbor(None)
        neighbor_depth = randint(clip_depth, self._MAX_DEPTH)
        divergence_cutoff: int = ceil(neighbor_depth * self._DIVERGENCE_CUTOFF_FACTOR)
        max_breadth = self._MAX_BREADTH
        neighbor = self._graph_generator.generate_random_graph_layers(
                clip_depth + 1,
                neighbor_depth,
                max_breadth,
                divergence_cutoff,
                clip_layers
        )
        if neighbor is None or len(neighbor.sinks) > 1:
            for _ in range(self._GRAPH_REGENERATION_THRESHOLD):
                self._logger.debug('Generated graph has more than one sink, re-generating neighbor')
                neighbor = self._create_neighbor(graph)
                if neighbor is not None:
                    return neighbor

            self._logger.debug('Generated graph has more than one sink, generating random graph')
            return self._create_complete_graph()

        return neighbor

    def _create_complete_graph(self) -> GraphOfOperations:
        graph_depth = randint(1, self._MAX_DEPTH)
        max_breadth = self._MAX_BREADTH
        divergence_cutoff: int = floor(graph_depth * self._DIVERGENCE_CUTOFF_FACTOR)
        return self._graph_generator.generate_random_graph(graph_depth, max_breadth, divergence_cutoff)