from ..api.controller import Controller
from ..api.controller.graph_of_operations_execution import GraphOfOperationsExecution
from ..api.graph.operation import OperationNode


class IterativeGraphController(Controller):
    """
    A controller for executing a graph of operations iteratively.
    The controller executes operations on an operation node basis.
    """

    def execute(self, operation_node: OperationNode, execution: GraphOfOperationsExecution) -> None:
        """
        Executes a given operation node in the context of an ongoing execution.
        :param operation_node: operation node to execute
        :param execution: ongoing execution
        """
        execution.process_operation(operation_node, self._process_operation_node)
