from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, ScoreExecOperation, \
    relative_complexity, absolute_complexity
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, Evaluator

op_split = PromptOperation(
        name='split',
        n_outputs=2,
        n_inputs=1,
        type=OperationType.GENERATE,
        output_complexity=relative_complexity(1, 2),
        prompt=Prompt(
                instruction='Split the given list into two lists of equal size. '
                            'Count the number of elements in the list before deciding where to split.'
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
        type=OperationType.AGGREGATE,
        output_complexity=relative_complexity(2),
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


def score_op_sum(cumulative_score: float, previous_state: State, current_state: State) -> float:
    """
    Determines the score of the sum operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :return: score
    """
    if cumulative_score < 0.0:
        return -1.0
    current_sum = current_state['sum'] if 'sum' in current_state else None
    previous_sum = (
        sum(previous_state['list']) if 'list' in previous_state
        else sum(
                sum(state_list) for state_list in previous_state['lists']
        ) if 'lists' in previous_state
        else None
    )
    if current_sum is not None and previous_sum is not None and current_sum == previous_sum:
        return 1.0
    return -1.0


op_sum = PromptOperation(
        name='sum',
        n_outputs=1,
        n_inputs=1,
        type=OperationType.GENERATE,
        output_complexity=absolute_complexity(1),
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
                type=OperationType.SCORE,
                score=score_op_sum,
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
