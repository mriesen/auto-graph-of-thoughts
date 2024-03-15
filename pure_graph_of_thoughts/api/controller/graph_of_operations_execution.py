from itertools import islice
from typing import Optional, Dict, Sequence, Callable

from ..graph.operation import GraphOfOperations, OperationNode
from ..graph.thought import GraphOfThoughts, ThoughtNode


class GraphOfOperationsExecution:
    """
    Represents an ongoing execution of a graph of operations.
    """

    _graph_of_operations: GraphOfOperations
    _graph_of_thoughts: GraphOfThoughts
    _operation_cursor: OperationNode
    _output_thought_nodes_by_operation_node: Dict[OperationNode, Sequence[ThoughtNode]]

    def __init__(
            self,
            graph_of_operations: GraphOfOperations,
            graph_of_thoughts: GraphOfThoughts,
            operation_cursor: Optional[OperationNode] = None
    ) -> None:
        self._graph_of_operations = graph_of_operations
        self._graph_of_thoughts = graph_of_thoughts
        self._operation_cursor = operation_cursor if operation_cursor is not None else graph_of_operations.source
        self._output_thought_nodes_by_operation_node = {}

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

    def get_output_thought_nodes(self, operation_node: OperationNode) -> Sequence[ThoughtNode]:
        return self._output_thought_nodes_by_operation_node[
            operation_node
        ] if operation_node in self._output_thought_nodes_by_operation_node else []

    def get_input_thought_nodes(self, operation_node: OperationNode) -> Sequence[ThoughtNode]:
        """
        Returns the input thought nodes for the given operation node.
        :param operation_node: operation node to get input thought nodes for
        :return: input thought nodes for operation node
        """
        if operation_node.is_source:
            return [
                self.graph_of_thoughts.source
            ]

        return [
            input_thought_node
            for predecessor in operation_node.predecessors
            for input_thought_node in
            self._get_input_thoughts_for_successor(predecessor, operation_node)
        ]

    def process_operation(
            self,
            operation_node: OperationNode,
            process: Callable[
                [OperationNode, Sequence[ThoughtNode]],
                Sequence[ThoughtNode]
            ]
    ) -> None:
        """
        Processes the given operation node.
        :param operation_node: the operation node to process
        :param process: the processing function to apply
        """
        self.operation_cursor = operation_node
        input_thought_nodes = self.get_input_thought_nodes(operation_node)
        output_thought_nodes: Sequence[ThoughtNode] = process(
                operation_node,
                input_thought_nodes
        )

        for input_thought_node in input_thought_nodes:
            input_thought_node.append_all(output_thought_nodes)

        self._output_thought_nodes_by_operation_node[operation_node] = output_thought_nodes

    def _get_input_thoughts_for_successor(
            self,
            predecessor_node: OperationNode,
            successor_node: OperationNode
    ) -> Sequence[ThoughtNode]:
        output_thought_nodes = self._output_thought_nodes_by_operation_node[
            predecessor_node
        ] if predecessor_node in self._output_thought_nodes_by_operation_node else []
        successors_input_thoughts = self._create_input_thoughts_buckets(
                predecessor_node.successors, output_thought_nodes
        )
        return successors_input_thoughts[predecessor_node.successors.index(successor_node)]

    @staticmethod
    def _create_input_thoughts_buckets(
            operation_nodes: Sequence[OperationNode], thoughts: Sequence[ThoughtNode]
    ) -> Sequence[Sequence[ThoughtNode]]:
        """
        Creates buckets of input thoughts for each operation node.
        Given a sequence of operation nodes and thoughts, the thoughts are filled into buckets
        of the size of the operation's number of inputs.
        Each subsequence represents the input thoughts for the operation of the node at the corresponding position
        in the given sequence of operation nodes.

        :param operation_nodes: operation nodes
        :param thoughts: thoughts to fill into buckets as input
        :return: buckets of input thoughts
        """
        n_inputs = [node.operation.n_inputs for node in operation_nodes]
        thoughts_iterator = iter(thoughts)
        return [list(islice(thoughts_iterator, n_input)) for n_input in n_inputs]
