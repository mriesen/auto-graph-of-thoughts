from auto_graph_of_thoughts.tasks.sum_list import op_sum, op_merge, op_split
from pure_graph_of_thoughts.api.graph.operation import OperationNode, GraphOfOperations


def _create_sum_list_graph_of_operations_8() -> GraphOfOperations:
    source = OperationNode.of(op_sum)
    graph_of_operations = GraphOfOperations.from_source(source)
    graph_of_operations.seal()
    return graph_of_operations


sum_list_graph_of_operations_8: GraphOfOperations = _create_sum_list_graph_of_operations_8()


def _create_sum_list_graph_of_operations_16() -> GraphOfOperations:
    source = OperationNode.of(op_split)
    for _ in (1, 2):
        source.append_operation(op_sum)
    aggregate = OperationNode.of(op_merge)
    for branch in source.successors:
        branch.append(aggregate)

    aggregate.append_operation(op_sum)

    graph_of_operations = GraphOfOperations.from_source(source)
    graph_of_operations.seal()
    return graph_of_operations


sum_list_graph_of_operations_16: GraphOfOperations = _create_sum_list_graph_of_operations_16()


def _create_sum_list_graph_of_operations_32() -> GraphOfOperations:
    source = OperationNode.of(op_split)
    aggregate_inner_nodes = []
    for _ in (1, 2):
        split = source.append_operation(op_split)
        for _ in (1, 2):
            split.append_operation(op_sum)
        aggregate_inner = OperationNode.of(op_merge)
        aggregate_inner_nodes.append(aggregate_inner)
        for branch in split.successors:
            branch.append(aggregate_inner)

    aggregate_outer = OperationNode.of(op_merge)
    for branch in aggregate_inner_nodes:
        branch.append(aggregate_outer)

    aggregate_outer.append_operation(op_sum)

    graph_of_operations = GraphOfOperations.from_source(source)
    graph_of_operations.seal()
    return graph_of_operations


sum_list_graph_of_operations_32: GraphOfOperations = _create_sum_list_graph_of_operations_32()
