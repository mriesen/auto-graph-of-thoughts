from typing import Optional, Dict, List, Sequence, Tuple

from ..graph.operation import GraphOfOperations, OperationNode
from ..graph.thought import GraphOfThoughts, ThoughtNode


class GraphOfOperationsExecution:
    """
    Represents an ongoing execution of a graph of operations.
    """

    _graph_of_operations: GraphOfOperations
    _graph_of_thoughts: GraphOfThoughts
    _operation_cursor: OperationNode
    _input_thought_nodes_by_operation_node: Dict[OperationNode, List[ThoughtNode]]

    def __init__(
            self,
            graph_of_operations: GraphOfOperations,
            graph_of_thoughts: GraphOfThoughts,
            operation_cursor: Optional[OperationNode] = None
    ) -> None:
        self._graph_of_operations = graph_of_operations
        self._graph_of_thoughts = graph_of_thoughts
        self._operation_cursor = operation_cursor if operation_cursor is not None else graph_of_operations.root
        self._input_thought_nodes_by_operation_node = {
            graph_of_operations.root: [graph_of_thoughts.root]
        }

    @property
    def graph_of_operations(self) -> GraphOfOperations:
        """The underlying graph of operations of the execution"""
        return self._graph_of_operations

    @property
    def graph_of_thoughts(self) -> GraphOfThoughts:
        """The generated graph of thoughts of the execution"""
        return self._graph_of_thoughts

    @property
    def operation_cursor(self) -> OperationNode:
        """The current operation in the graph of operations"""
        return self._operation_cursor

    @operation_cursor.setter
    def operation_cursor(self, operation_node: OperationNode) -> None:
        """
        Sets the operation cursor to the given operation node.
        :param operation_node: operation node to set as cursor
        """
        self._operation_cursor = operation_node

    @property
    def current_input_thought_nodes(self) -> Sequence[ThoughtNode]:
        """The input thought nodes for the current operation node"""
        return self._input_thought_nodes_by_operation_node[self._operation_cursor]

    def update_input_thoughts(
            self,
            input_thought_nodes_by_operation_node: Sequence[Tuple[OperationNode, Sequence[ThoughtNode]]]
    ) -> None:
        """
        Updates the input thought nodes by operation node.
        :param input_thought_nodes_by_operation_node: mappings of an operation node to thought nodes
        """
        for operation_node, input_thought_nodes in input_thought_nodes_by_operation_node:
            if operation_node not in self._input_thought_nodes_by_operation_node:
                self._input_thought_nodes_by_operation_node[operation_node] = []
            self._input_thought_nodes_by_operation_node[operation_node].extend(input_thought_nodes)
