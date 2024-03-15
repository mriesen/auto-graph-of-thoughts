from ..api.graph.operation import OperationNode, GraphOfOperations
from ..api.language_model import Prompt, Example
from ..api.operation import PromptOperation, OperationType, ScoreExecOperation, OperationRegistry, Evaluator

op_split = PromptOperation(
        name='split',
        n_outputs=2,
        n_inputs=1,
        type=OperationType.generate,
        prompt=Prompt(
                instruction='Split the given list into two lists of equal size. '
                            'Only output the lists in JSON format as the examples show.',
                examples=[
                    Example(
                            input={
                                'list': [8, 2, 0, 1]
                            },
                            output={
                                'lists': [
                                    [8, 2],
                                    [0, 1]
                                ]
                            }
                    )
                ],
        ),
        transform_after=lambda state: [{'list': state_list} for state_list in state['lists']]
)

op_merge = PromptOperation(
        name='merge',
        n_outputs=1,
        n_inputs=2,
        type=OperationType.aggregate,
        prompt=Prompt(
                instruction='Combine the given lists to a single list nested in another list. '
                            'Only output the list in JSON format as the examples show.',
                examples=[
                    Example(
                            input={
                                'lists': [
                                    [8, 2],
                                    [0, 1]
                                ]
                            },
                            output={
                                'list': [8, 2, 0, 1]
                            }
                    )
                ]
        ),
        transform_before=lambda states: {
            'lists': [
                item for item in [
                    state['list'] if 'list' in state
                    else [state['sum']] if 'sum' in state
                    else None for state in states
                ] if item is not None
            ]
        }
)

op_sum = PromptOperation(
        name='generate_single',
        n_outputs=1,
        n_inputs=1,
        type=OperationType.generate,
        prompt=Prompt(
                instruction='Calculate the sum of the provided list and output the sum '
                            'in JSON format like the examples show.',
                examples=[
                    Example(
                            input={
                                'list': [8, 2, 0, 1]
                            },
                            output={
                                'sum': 11
                            }
                    )
                ]
        ),
        score_operation=ScoreExecOperation(
                name='score',
                type=OperationType.score,
                score=lambda previous_state, current_state: float(
                        (
                            sum(previous_state['list']) if 'list' in previous_state
                            else sum(
                                    state_list for state_list in previous_state['lists']
                            ) if 'lists' in previous_state
                            else 0
                        ) == current_state['sum']
                ),
                n_inputs=1,
                n_outputs=1
        )
)

sum_list_registry = OperationRegistry(
        operations=[op_sum, op_split, op_merge],
        evaluator=Evaluator(lambda initial_state, state: sum(initial_state['list']) == state['sum'])
)


def _create_sum_list_graph_of_operations_8() -> GraphOfOperations:
    return GraphOfOperations.from_source(OperationNode.of(op_sum))


sum_list_graph_of_operations_8: GraphOfOperations = _create_sum_list_graph_of_operations_8()


def _create_sum_list_graph_of_operations_16() -> GraphOfOperations:
    source = OperationNode.of(op_split)
    for _ in (1, 2):
        source.append_operation(op_sum)
    aggregate = OperationNode.of(op_merge)
    for branch in source.successors:
        branch.append(aggregate)

    aggregate.append_operation(op_sum)

    return GraphOfOperations.from_source(source)


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

    return GraphOfOperations.from_source(source)


sum_list_graph_of_operations_32: GraphOfOperations = _create_sum_list_graph_of_operations_32()
