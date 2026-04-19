from math import ceil
from typing import Sequence, Optional, Callable, Tuple, List, Dict

from pure_graph_of_thoughts.api.controller import Controller, GraphOfOperationsExecution, ControllerException
from pure_graph_of_thoughts.api.graph import GraphMutationException
from pure_graph_of_thoughts.api.graph.operation import GraphOfOperations, OperationNode
from pure_graph_of_thoughts.api.graph.thought import GraphOfThoughts
from pure_graph_of_thoughts.api.language_model import LanguageModel
from pure_graph_of_thoughts.api.operation import Operation, Complexity, AbsoluteComplexity, RelativeComplexity
from pure_graph_of_thoughts.api.state import State
from .layer_action_result import LayerActionResult


class ContinuousGraphController(Controller):
    """
    A controller for executing a graph of operations on a layer basis.
    """

    _generate_init_state: Callable[[], Tuple[int, State]]

    _complexity: int

    _local_complexity: int

    _local_complexities: Dict[int, int]

    _init_state: State

    _execution: Optional[GraphOfOperationsExecution]

    _max_depth: int

    _max_breadth: int

    _divergence_cutoff_factor: float

    _max_complexity: int

    _max_operations: int

    _n_operations: int

    @property
    def max_depth(self) -> int:
        """The maximum depth"""
        return self._max_depth

    @property
    def max_breadth(self) -> int:
        """The maximum breadth"""
        return self._max_breadth

    @property
    def max_complexity(self) -> int:
        """The maximum complexity"""
        return self._max_complexity

    @property
    def max_operations(self) -> int:
        """The maximum number of operations"""
        return self._max_operations


    @property
    def divergence_cutoff(self) -> int:
        """The divergence cutoff"""
        return ceil(self._max_depth * self._divergence_cutoff_factor)

    @property
    def is_initialized(self) -> bool:
        """Whether the controller is initialized"""
        return self._execution is not None

    @property
    def init_state(self) -> State:
        """The initial state"""
        return self._init_state

    @property
    def graph_of_thoughts(self) -> Optional[GraphOfThoughts]:
        """The graph of thoughts"""
        if self._execution is None:
            return None
        return self._execution.graph_of_thoughts

    @property
    def graph_of_operations(self) -> GraphOfOperations:
        """The graph of operations"""
        if self._execution is None:
            raise ControllerException('Graph of operations is None')
        return self._execution.graph_of_operations

    @property
    def current_depth(self) -> int:
        """The current depth"""
        return self._execution.graph_of_operations.depth if self._execution is not None else 0

    @property
    def current_breadth(self) -> int:
        """The current breadth"""
        return self._execution.graph_of_operations.breadth if self._execution is not None else 0

    @property
    def complexity(self) -> int:
        """The complexity"""
        return self._complexity

    @property
    def local_complexity(self) -> int:
        """The local complexity"""
        return self._local_complexity

    @property
    def divergence(self) -> bool:
        """Whether divergence is allowed"""
        return self.current_depth <= self.divergence_cutoff

    @property
    def n_operations(self) -> int:
        """The number of executed operations"""
        return self._n_operations

    @property
    def _present_execution(self) -> GraphOfOperationsExecution:
        if self._execution is None:
            raise ControllerException('Execution is None')
        return self._execution

    def __init__(
            self,
            language_model: LanguageModel,
            generate_init_state: Callable[[], Tuple[int, State]],
            max_depth: int,
            max_breadth: int,
            divergence_cutoff_factor: float,
            max_complexity: int,
            max_operations: int
    ) -> None:
        super().__init__(language_model)
        self._generate_init_state = generate_init_state
        self._max_depth = max_depth
        self._max_breadth = max_breadth
        self._divergence_cutoff_factor = divergence_cutoff_factor
        self._max_complexity = max_complexity
        self._max_operations = max_operations

        self._execution = None
        self._complexity, self._init_state = self._generate_init_state()
        self._local_complexity = self._complexity
        self._local_complexities = {}
        self._n_operations = 0

    def append_layer(self, operation: Operation) -> LayerActionResult:
        """
        Appends a layer of a given operation to the graph of operations.
        :param operation: operation to append layer of
        :return: action result
        """

        if not self.is_initialized:

            # first operation must have a single input
            if operation.n_inputs != 1:
                return LayerActionResult.invalid()

            local_complexity = self._calculate_local_complexity(operation.output_complexity)
            if local_complexity > self._max_complexity:
                return LayerActionResult.invalid()

            self._local_complexity = local_complexity

            # process valid first operation
            self._initialize_controller(operation)
            self._execute_sink_layer()

            self._local_complexities[self.graph_of_operations.sink_layer_index] = self._local_complexity
            return self._create_layer_action_result()

        predecessor_n_outputs, n_operations = self._prepare_append_operation(operation)
        is_valid = self._validate_append_operation(operation, predecessor_n_outputs, n_operations)

        if not is_valid:
            return LayerActionResult.invalid()

        operation_nodes: Sequence[OperationNode] = [OperationNode.of(operation) for _ in range(n_operations)]

        local_complexity = self._calculate_local_complexity(operation.output_complexity)
        if local_complexity > self._max_complexity:
            return LayerActionResult.invalid()

        self._local_complexity = local_complexity
        self.graph_of_operations.append_layer(operation_nodes)
        self._execute_sink_layer()
        self._local_complexities[self.graph_of_operations.sink_layer_index] = self._local_complexity
        return self._create_layer_action_result()

    def _prepare_append_operation(self, operation: Operation) -> Tuple[int, int]:
        """
        Calculates the number of outputs of the predecessors and the number of operations to append.
        :param operation: operation to append layer of
        :return: tuple of number of outputs of the predecessors and number of operations to append
        """
        operation_array = self.graph_of_operations.operation_array
        predecessor_n_outputs = sum([operation.n_outputs for operation in operation_array[-1]])
        n_operations: int = predecessor_n_outputs // operation.n_inputs
        return predecessor_n_outputs, n_operations

    def _calculate_local_complexity(self, output_complexity: Complexity) -> int:
        if isinstance(output_complexity, AbsoluteComplexity):
            return output_complexity
        elif isinstance(output_complexity, RelativeComplexity):
            return max(round(self._local_complexity * output_complexity), 1)

    def validate_append_operation(self, operation: Operation) -> bool:
        """
        Validates the appending of an operation.
        :param operation: operation to append layer of
        :return: whether the appending of the operation is valid
        """
        if not self.is_initialized:
            return operation.n_inputs == 1
        predecessor_n_outputs, n_operations = self._prepare_append_operation(operation)
        return self._validate_append_operation(operation, predecessor_n_outputs, n_operations)

    def _validate_append_operation(self, operation: Operation, predecessor_n_outputs: int, n_operations: int) -> bool:

        # honor max operations
        if self.n_operations >= self.max_operations:
            return False

        # honor max depth
        if self.current_depth > self._max_depth - 1:
            return False

        # number of predecessors' outputs must be dividable by number of inputs
        if predecessor_n_outputs % operation.n_inputs != 0:
            return False

        # honor max breadth
        if operation.n_outputs * n_operations > self._max_breadth:
            return False

        # honor divergence
        if not self.divergence and operation.n_outputs > 1:
            return False

        return True

    def remove_sink_layer(self) -> LayerActionResult:
        """
        Removes the sink layer of the graph of operations.
        :return: action result
        """
        if not self.is_initialized or self.graph_of_operations.sink_layer_index == 0:
            return LayerActionResult.invalid()
        graph_of_operations = self.graph_of_operations
        graph_of_thoughts = self.graph_of_thoughts
        try:
            graph_of_operations.remove_layer(graph_of_operations.sink_layer_index)
            if graph_of_thoughts is not None:
                graph_of_thoughts.remove_layer(graph_of_thoughts.sink_layer_index)
            self._local_complexity = self._local_complexities[graph_of_operations.sink_layer_index]
            return self._create_layer_action_result()
        except GraphMutationException:
            return LayerActionResult.invalid()

    def _create_layer_action_result(self) -> LayerActionResult:
        sink_thought_layer = self._present_execution.graph_of_thoughts.sink_layer
        sink_thoughts = [thought_node.thought for thought_node in sink_thought_layer]

        scores = [
            sink_thought.score for sink_thought in sink_thoughts if sink_thought.score is not None
        ]
        score = min(scores) if len(scores) > 0 else None

        return LayerActionResult(score=score)

    def _execute_sink_layer(self) -> None:
        sink_operation_layer: Sequence[OperationNode] = self.graph_of_operations.sink_layer
        self._n_operations += len(sink_operation_layer)
        for operation_node in sink_operation_layer:
            self._present_execution.process_operation(operation_node, self._process_operation_node)

    def _initialize_controller(self, operation: Operation) -> None:
        source_operation_node = OperationNode.of(operation)
        graph_of_operations = GraphOfOperations.from_source(source_operation_node)
        graph_of_thoughts = GraphOfThoughts.from_init_state(self._init_state)
        self._execution = GraphOfOperationsExecution(
                graph_of_operations=graph_of_operations,
                graph_of_thoughts=graph_of_thoughts,
                operation_cursor=source_operation_node
        )

    def reset(self) -> None:
        """
        Resets the execution state.
        """
        self._execution = None
        self._complexity, self._init_state = self._generate_init_state()
        self._local_complexity = self._complexity
        self._local_complexities = {}
        self._n_operations = 0
