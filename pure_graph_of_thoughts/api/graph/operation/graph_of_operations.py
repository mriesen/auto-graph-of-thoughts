import copy
from dataclasses import dataclass
from typing import Self, Sequence, Dict, Any, List, Mapping

from .operation_matrix import OperationMatrix
from .operation_node import OperationNode
from .operation_node_schema import OperationNodeSchema
from ..graph import Graph
from ..graph_schema import GraphSchema
from ...operation import Operation, OperationKey


@dataclass
class GraphOfOperations(Graph[OperationNode]):
    """
    Represents a graph of operations.
    """

    @property
    def operation_matrix(self) -> OperationMatrix:
        """
        Returns the operation matrix of the graph.
        :return: operation matrix
        """
        return [
            [node.operation for node in layer]
            for layer in self.layers
        ]

    @classmethod
    def from_operation_matrix(cls, operation_matrix: OperationMatrix) -> Self:
        """
        Creates a graph of operations from a given operation matrix.
        :param operation_matrix: operation matrix
        :return: created graph of operations
        """

        source_operation: Operation = operation_matrix[0][0]
        source_node: OperationNode = OperationNode.of(source_operation)
        predecessor_nodes: Sequence[OperationNode] = [source_node]
        for operation_layer in operation_matrix[1:]:
            successor_nodes: Sequence[OperationNode] = [
                OperationNode.of(operation) for operation in operation_layer
            ]
            predecessor_nodes = cls._connect_layer(predecessor_nodes, successor_nodes)
        return cls.from_source(source_node)

    def __deepcopy__(self, memo: Dict[Any, Any]) -> Self:
        return self.from_operation_matrix(self.operation_matrix)

    def clone(self) -> Self:
        """
        Clones the current graph of operations and returns a deep copy of the graph of operations.
        :return: deep copy of the operations
        """
        return copy.deepcopy(self)

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
        assert len(predecessor_nodes_by_output_index) == len(successor_nodes_by_input_index), \
            'The number of outputs of the predecessor nodes must match the number of inputs of the successor nodes'

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
