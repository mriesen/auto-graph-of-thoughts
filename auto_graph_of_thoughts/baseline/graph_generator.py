import logging
from abc import ABC
from itertools import product
from random import choice
from typing import Sequence, Mapping, Set, Dict, Optional, Callable

from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations
from pure_graph_of_thoughts.api.graph.operation.operation_matrix import OperationMatrix
from pure_graph_of_thoughts.api.operation import Operation


class GraphGenerator(ABC):
    """
    Generator for graph of operations.
    """

    _operations: Sequence[Operation]
    _operations_by_n_inputs: Mapping[int, Set[Operation]]
    _logger: logging.Logger

    def __init__(self, operations: Sequence[Operation]) -> None:
        self._operations = operations
        self._operations_by_n_inputs = self._create_n_inputs_operations_mapping(operations)
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate_random_graph(self, graph_depth: int, max_breadth: int, divergence_cutoff: int) -> GraphOfOperations:
        """
        Generates a random graph of operations.
        :param graph_depth: desired depth of the graph
        :param max_breadth: maximum breadth of the graph
        :param divergence_cutoff: divergence cutoff depth
        :return: generated graph of operations
        """
        single_input_operations = list(self._operations_by_n_inputs[1])
        source_operation = choice(single_input_operations)
        operation_matrix: OperationMatrix = [
            [source_operation]
        ]
        graph_of_operations = self.generate_random_graph_layers(
                1,
                graph_depth,
                max_breadth,
                divergence_cutoff,
                operation_matrix
        )
        if graph_of_operations is None:
            self._logger.debug('Generated graph has more than one sink, re-generating')
            return self.generate_random_graph(graph_depth, max_breadth, divergence_cutoff)

        return graph_of_operations

    def generate_random_graph_layers(
            self,
            depth_start: int,
            depth_end: int,
            max_breadth: int,
            divergence_cutoff: int,
            initial_operation_matrix: OperationMatrix
    ) -> Optional[GraphOfOperations]:
        """
        Generates random graph layers.
        :param depth_start: start depth
        :param depth_end: end depth
        :param max_breadth: maximum breadth of the graph
        :param divergence_cutoff: divergence cutoff depth
        :param initial_operation_matrix: initial operation matrix
        :return: generated graph of operations
        """

        operation_matrix: OperationMatrix = initial_operation_matrix
        for depth in range(depth_start, depth_end):
            divergence = depth <= divergence_cutoff
            expect_single_output = depth == depth_end
            operation_matrix_result: Optional[OperationMatrix] = self._generate_next_layer(
                    operation_matrix,
                    max_breadth,
                    divergence,
                    expect_single_output
            )
            if operation_matrix_result is None:
                return None
            operation_matrix = operation_matrix_result
        return GraphOfOperations.from_operation_matrix(operation_matrix)

    def _generate_next_layer(
            self,
            operation_matrix: OperationMatrix,
            max_breadth: int,
            divergence: bool,
            expect_single_output: bool
    ) -> Optional[OperationMatrix]:

        predecessor_operations: Sequence[Operation] = operation_matrix[-1]

        successor_operation_candidates = self._next_operation_candidates(
                predecessor_operations,
                max_breadth,
                divergence,
                expect_single_output
        )
        if len(successor_operation_candidates) == 0:
            return None
        successor_operations: Sequence[Operation] = choice(successor_operation_candidates)

        return list(operation_matrix) + [successor_operations]

    def _next_operation_candidates(
            self,
            predecessors: Sequence[Operation],
            max_breadth: int,
            divergence: bool = True,
            expect_single_output: bool = False
    ) -> Sequence[Sequence[Operation]]:
        n_total_inputs = sum([predecessor.n_outputs for predecessor in predecessors])
        n_inputs_combinations = self._get_n_inputs_combinations(n_total_inputs)
        next_operation_candidates = [
            list(operations) for n_inputs in n_inputs_combinations for operations in product(*[
                self._operations_by_n_inputs[n_input] for n_input in n_inputs if n_input in self._operations_by_n_inputs
            ]) if (
                    len(operations) > 0
                    and sum([operation.n_inputs for operation in operations]) == n_total_inputs
                    and sum([operation.n_outputs for operation in operations]) <= max_breadth
            )
        ]

        filter_operations: Optional[Callable[[Sequence[Operation]], bool]] = None
        if expect_single_output:
            filter_operations = lambda operations: sum([operation.n_outputs for operation in operations]) == 1
        elif not divergence:
            filter_operations = lambda operations: not any(operation.n_outputs > 1 for operation in operations)

        if filter_operations is not None:
            return list(
                    filter(
                            filter_operations,
                            next_operation_candidates
                    )
            )

        return next_operation_candidates

    @staticmethod
    def _get_n_inputs_combinations(max_n_inputs: int) -> Sequence[Sequence[int]]:
        # TODO: investigate expensive cartesian product generation
        return [
            list(n) for n_inputs in range(1, max_n_inputs, 1)
            for n in product(range(1, max_n_inputs + 1), repeat=n_inputs) if sum(n) == max_n_inputs
        ] + [
            [1 for _ in range(max_n_inputs)]
        ]

    @staticmethod
    def _create_n_inputs_operations_mapping(operations: Sequence[Operation]) -> Mapping[int, Set[Operation]]:
        n_inputs_operations: Dict[int, Set[Operation]] = {}
        for operation in operations:
            n_inputs = operation.n_inputs
            if n_inputs not in n_inputs_operations:
                n_inputs_operations[n_inputs] = set()
            n_inputs_operations[n_inputs].add(operation)
        return n_inputs_operations
