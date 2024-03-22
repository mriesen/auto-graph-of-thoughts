from collections import deque
from typing import Deque, Set

from ..api.controller.controller import Controller
from ..api.controller.graph_of_operations_execution import GraphOfOperationsExecution
from ..api.graph.operation import GraphOfOperations, OperationNode
from ..api.graph.thought import GraphOfThoughts
from ..api.language_model import LanguageModel
from ..api.state import State


class CompleteGraphController(Controller):
    """
    A controller for executing a complete graph of operations.
    The controller traverses a given graph of operations with a breadth-first approach
    and executes all operations in order.
    """

    def __init__(self, language_model: LanguageModel) -> None:
        """
        Initializes a new complete graph controller instance.
        :param language_model: language model to use
        """
        super().__init__(language_model)

    def execute_graph(self, graph_of_operations: GraphOfOperations, init_state: State) -> GraphOfThoughts:
        """
        Executes a graph of operations.
        Traverses the graph breadth-first and executes each node's operation.
        If applicable, the scoring operation is performed afterward.

        :param graph_of_operations: graph of operations
        :param init_state: initial state
        :return: all thoughts by nodes
        """
        visited: Set[OperationNode] = set()
        queue: Deque[OperationNode] = deque([graph_of_operations.source])
        graph_of_thoughts: GraphOfThoughts = GraphOfThoughts.from_init_state(init_state)
        execution = GraphOfOperationsExecution(graph_of_operations, graph_of_thoughts)

        while queue:
            operation_node: OperationNode = queue.popleft()
            if operation_node not in visited:
                self._logger.info('Traversing node %s', operation_node.id)

                execution.process_operation(operation_node, self._process_operation_node)

                visited.add(operation_node)

            queue.extend([successor for successor in operation_node.successors if successor not in visited])

        graph_of_thoughts.seal()
        return graph_of_thoughts
