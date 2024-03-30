import copy
from dataclasses import dataclass
from typing import Self, Sequence, Dict, Any, List, Mapping

from .operation_array import OperationArray
from .operation_graph_schema import OperationGraphSchema
from .operation_node import OperationNode
from .operation_node_schema import OperationNodeSchema
from ..graph import Graph
from ..graph_schema import GraphSchema
from ...internal.seal import mutating, MutationScope
from ...operation import Operation, OperationKey


@dataclass
class GraphOfOperations(Graph[OperationNode, OperationGraphSchema]):
    """
    Represents a graph of operations.
    """

    @property
    def operation_array(self) -> OperationArray:
        """
        Returns the operation array of the graph.
        :return: operation array
        """
        return [
            [node.operation for node in layer]
            for layer in self.layers
        ]

    @classmethod
    def from_operation_array(cls, operation_array: OperationArray) -> Self:
        """
        Creates a graph of operations from a given operation array.
        :param operation_array: operation array
        :return: created graph of operations
        """

        source_operation: Operation = operation_array[0][0]
        source_node: OperationNode = OperationNode.of(source_operation)
        predecessor_nodes: Sequence[OperationNode] = [source_node]
        for operation_layer in operation_array[1:]:
            successor_nodes: Sequence[OperationNode] = [
                OperationNode.of(operation) for operation in operation_layer
            ]
            predecessor_nodes = cls._connect_layer(predecessor_nodes, successor_nodes)
        return cls.from_source(source_node)

    @mutating(scope=MutationScope.SELF)
    def append_layer(self, successor_nodes: Sequence[OperationNode]) -> None:
        """
        Appends a layer of nodes to the current graph.
        :param successor_nodes: successor nodes to append
        """
        predecessor_nodes: Sequence[OperationNode] = self.layers[-1]
        self._connect_layer(predecessor_nodes, successor_nodes)

    @mutating(scope=MutationScope.SELF)
    def remove_layer(self, depth: int) -> None:
        """
        Removes the layer at the given depth from the graph.
        This causes the removal of all successive layers as well.
        :param depth: depth to remove
        """
        if depth == 0:
            raise GraphOfOperationsException('Cannot remove layer at depth 0')
        if depth > self.depth:
            raise GraphOfOperationsException('Layer to remove at the given depth does not exist')
        operation_nodes: Sequence[OperationNode] = self.layers[depth - 1]
        for operation_node in operation_nodes:
            operation_node.remove_all_successors()

    def __deepcopy__(self, memo: Dict[Any, Any]) -> Self:
        operations: Sequence[Operation] = [node.operation for node in self.nodes]
        return self.from_schema(self.to_schema(), operations)

    def to_schema(self) -> OperationGraphSchema:
        return super()._to_schema(OperationGraphSchema)

    @staticmethod
    def _connect_layer(
            predecessor_nodes: Sequence[OperationNode], successor_nodes: Sequence[OperationNode]
    ) -> Sequence[OperationNode]:
        """
        Connects the given successor nodes with the given predecessor nodes.
        :param predecessor_nodes: predecessor nodes
        :param successor_nodes: successor nodes
        :return: connected layer
        """
        predecessor_nodes_by_output_index: List[OperationNode] = [
            predecessor_node
            for predecessor_node in predecessor_nodes
            for _ in range(predecessor_node.operation.n_outputs)
        ]
        successor_nodes_by_input_index: List[OperationNode] = [
            successor_node
            for successor_node in successor_nodes
            for _ in range(successor_node.operation.n_inputs)
        ]
        if len(predecessor_nodes_by_output_index) != len(successor_nodes_by_input_index):
            raise GraphOfOperationsException(
                    'The number of outputs of the predecessor nodes '
                    'must match the number of inputs of the successor nodes'
            )

        for i, successor_node in enumerate(successor_nodes_by_input_index):
            predecessor_nodes_by_output_index[i].append(successor_node)

        return successor_nodes

    @classmethod
    def from_schema(cls, schema: GraphSchema[OperationNodeSchema], operations: Sequence[Operation]) -> Self:
        """
        Constructs a graph of operations from a given graph schema and a sequence of operations.
        :param schema: graph schema
        :param operations: operations
        :return: constructed graph of operations
        """
        operations_by_key: Mapping[OperationKey, Operation] = {
            operation.key: operation for operation in operations
        }
        nodes = [
            OperationNode.of(
                    id=node.id,
                    operation=operations_by_key[node.operation_key]
            ) for node in schema.nodes
        ]
        return cls._construct_graph(nodes, schema.edges)


class GraphOfOperationsException(Exception):
    """
    An exception raised in context of a graph of operations.
    """

    def __init__(self, message: str):
        super().__init__(message)
