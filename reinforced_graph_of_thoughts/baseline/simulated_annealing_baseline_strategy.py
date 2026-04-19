from math import exp, floor
from typing import Optional, List

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from .baseline_strategy import BaselineStrategy
from .model import BaselineIterationResult, BaselineResultSummary
from .simulated_annealing_baseline_config import SimulatedAnnealingBaselineConfig


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

    _META_INFO_TEMPERATURE = 'temperature'
    _META_INFO_ENERGY = 'energy'
    _META_INFO_DELTA_ENERGY = 'delta_energy'
    _META_INFO_PROBABILITY = 'probability'
    _META_INFO_IS_SELECTED = 'is_selected'

    _temperature: float
    _cooling_factor: float
    _best_energy: float
    _all_results: List[BaselineIterationResult]
    _selected_result: Optional[BaselineIterationResult]
    _neighbor_regeneration_threshold: int
    _max_cost: int

    @property
    def current_result(self) -> Optional[BaselineIterationResult]:
        """The current result"""
        return self._selected_result

    @property
    def _selected_energy(self) -> float:
        if self._selected_result is None:
            return 1.0
        return self._calculate_energy(self._selected_result)

    def __init__(self, config: SimulatedAnnealingBaselineConfig) -> None:
        """
        Instantiates a simulated annealing baseline strategy.
        :param config: baseline strategy configuration
        """
        super().__init__(config)

        self._temperature = 1.0
        self._cooling_factor = config.cooling_factor
        self._best_energy = 0
        self._all_results = []
        self._selected_result = None
        self._neighbor_regeneration_threshold = config.neighbor_regeneration_threshold
        self._max_cost = config.max_depth * config.max_breadth

    def generate(self, max_iterations: int, stop_on_first_valid: bool = False) -> BaselineResultSummary:
        for i in range(1, max_iterations + 1):
            iteration_result = self._generate_single(i)
            self._all_results.append(iteration_result)
            if stop_on_first_valid and iteration_result.is_valid:
                return BaselineResultSummary(
                        results=self._all_results,
                        final_result_index=self._all_results.index(iteration_result),
                        max_iterations=max_iterations,
                        stop_on_first_valid=True
                )

        return BaselineResultSummary(
                results=self._all_results,
                final_result_index=self._all_results.index(
                        self._selected_result
                ) if self._selected_result is not None else None,
                max_iterations=max_iterations
        )

    def _generate_single(self, iteration: int) -> BaselineIterationResult:
        self._temperature *= self._cooling_factor
        neighbor_graph = self._create_neighbor(
                GraphOfOperations.from_schema(
                        self._selected_result.graph_of_operations,
                        self._operations
                ) if self._selected_result is not None and self._selected_result.is_valid else None
        )
        iteration_result = self._evaluate_graph(neighbor_graph, iteration)
        current_energy = self._calculate_energy(iteration_result)
        selected_energy = self._selected_energy
        delta_energy = current_energy - selected_energy
        probability: float = exp(-delta_energy / self._temperature)

        is_selected = delta_energy < 0 or self._random.random() <= probability

        iteration_result = iteration_result.with_meta_info({
            self._META_INFO_ENERGY: current_energy,
            self._META_INFO_DELTA_ENERGY: delta_energy,
            self._META_INFO_IS_SELECTED: is_selected,
            self._META_INFO_TEMPERATURE: self._temperature,
            self._META_INFO_PROBABILITY: probability
        })

        if is_selected:
            self._selected_result = iteration_result

        return iteration_result

    def _calculate_energy(self, iteration_result: BaselineIterationResult) -> float:
        """
        Calculates the energy of a given iteration result.
        The energy is 1 if the iteration result is not valid.
        Otherwise, the energy corresponds to cost / max_cost.

        :param iteration_result: iteration result
        :return: energy
        """
        if not iteration_result.is_valid:
            return 1.0
        return iteration_result.cost / self._max_cost

    def _create_neighbor(self, graph: Optional[GraphOfOperations] = None) -> GraphOfOperations:
        if graph is None:
            return self._create_complete_graph()

        operation_array = graph.operation_array

        prev_depth = len(operation_array)

        clip_depth = self._random.randint(0, prev_depth)
        clip_layers = operation_array[:clip_depth - 1]

        if len(clip_layers) == 0:
            return self._create_neighbor()
        neighbor_depth = self._random.randint(clip_depth, self._config.max_depth)
        divergence_cutoff: int = floor(neighbor_depth * self._config.max_depth)
        max_breadth = self._config.max_breadth
        neighbor = self._graph_generator.generate_random_graph_layers(
                clip_depth + 1,
                neighbor_depth,
                max_breadth,
                divergence_cutoff,
                clip_layers
        )
        if neighbor is None or len(neighbor.sinks) > 1:
            for _ in range(self._neighbor_regeneration_threshold):
                self._logger.debug('Generated graph has more than one sink, re-generating neighbor')
                neighbor = self._create_neighbor(graph)
                if neighbor is not None:
                    return neighbor

            self._logger.debug('Generated graph has more than one sink, generating random graph')
            return self._create_complete_graph()

        return neighbor

    def _create_complete_graph(self) -> GraphOfOperations:
        graph_depth = self._random.randint(1, self._config.max_depth)
        max_breadth = self._config.max_breadth
        divergence_cutoff: int = floor(graph_depth * self._config.divergence_cutoff_factor)
        return self._graph_generator.generate_random_graph(graph_depth, max_breadth, divergence_cutoff)
