from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, ScoreExecOperation
from pure_graph_of_thoughts.api.task import Task, Evaluator

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
        transform_before=lambda states: {
            'list': []
        } if len(states) == 0 else {
            'list': [states[0]['sum']]
        } if 'sum' in states[0] else states[0],
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
        transform_before=lambda states: {
            'list': []
        } if len(states) == 0 else {
            'list': [states[0]['sum']]
        } if 'sum' in states[0] else states[0],
        score_operation=ScoreExecOperation(
                name='score',
                type=OperationType.score,
                score=lambda previous_state, current_state: float(
                        'sum' in current_state and (
                            sum(previous_state['list']) if 'list' in previous_state
                            else sum(
                                    sum(state_list) for state_list in previous_state['lists']
                            ) if 'lists' in previous_state
                            else 0
                        ) == current_state['sum']
                ),
                n_inputs=1,
                n_outputs=1
        )
)

sum_list_task = Task(
        operations=[op_sum, op_split, op_merge],
        evaluator=Evaluator(
                lambda initial_state, state: 'list' in initial_state
                                             and 'sum' in state
                                             and sum(initial_state['list']) == state['sum']
        )
)
