from typing import Sequence

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, ScoreExecOperation, \
    relative_complexity, absolute_complexity
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, Evaluator


def validate_op_split(previous_state: State, current_state: State, output_states: Sequence[State]) -> bool:
    """
    Checks whether the split operation was performed correctly.
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return:
    """
    min_tolerated_sublist_ratio = 0.5
    if 'list' in previous_state and len(output_states) == 2 and 'list' in output_states[0] and 'list' in output_states[
        1]:
        previous_list: Sequence[int] = previous_state['list']
        output_lists: Sequence[Sequence[int]] = [output_state['list'] for output_state in output_states]
        sum_correct = sum(sum(l) for l in output_lists) == sum(previous_list)
        half_len_previous_list = len(previous_list) / 2 if len(previous_list) > 0 else 0
        sublist_ratio = 1 - (abs(len(output_lists[0]) - len(output_lists[1])) / (half_len_previous_list if half_len_previous_list > 0 else 1.0))
        return sum_correct and sublist_ratio >= min_tolerated_sublist_ratio
    return False


def score_op_split(cumulative_score: float, previous_state: State, current_state: State,
                   output_states: Sequence[State]) -> float:
    """
    Determines the score of the split operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return: score
    """
    if cumulative_score < 0.0:
        return -1.0
    if validate_op_split(previous_state, current_state, output_states):
        return 1.0
    return -1.0


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
    score_operation=ScoreExecOperation(
        name='score_split',
        type=OperationType.SCORE,
        score=score_op_split,
        n_inputs=1,
        n_outputs=1
    ),
    transform_before=lambda states: {
        'list': []
    } if len(states) == 0 else {
        'list': [states[0]['sum']]
    } if 'sum' in states[0] else states[0],
    transform_after=lambda state: [{'list': state_list} for state_list in state['lists']]
)

def validate_op_merge(previous_state: State, current_state: State, output_states: Sequence[State]) -> bool:
    """
    Checks whether the merge operation was performed correctly.
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return:
    """
    if 'lists' in previous_state and 'list' in current_state and len(previous_state['lists']) == 2:
        previous_lists: Sequence[Sequence[int]] = previous_state['lists']
        current_list: Sequence[int] = current_state['list']
        sum_correct = sum(sum(l) for l in previous_lists) == sum(current_list)
        return sum_correct and len(current_list) == len(previous_lists[0]) + len(previous_lists[1])
    return False

def score_op_merge(cumulative_score: float, previous_state: State, current_state: State, output_states: Sequence[State]) -> float:
    """
    Determines the score of the split operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return: score
    """
    if cumulative_score < 0.0:
        return -1.0
    if validate_op_merge(previous_state, current_state, output_states):
        return 1.0
    return -1.0

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
    score_operation=ScoreExecOperation(
        name='score_merge',
        type=OperationType.SCORE,
        score=score_op_merge,
        n_inputs=1,
        n_outputs=1
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


def validate_op_sum(previous_state: State, current_state: State) -> bool:
    """
    Checks whether the sum operation was performed correctly.
    :param previous_state: previous state
    :param current_state: current state
    :return:
    """
    current_sum = current_state['sum'] if 'sum' in current_state else None
    previous_sum = (
        sum(previous_state['list']) if 'list' in previous_state
        else sum(
            sum(state_list) for state_list in previous_state['lists']
        ) if 'lists' in previous_state
        else None
    )
    return current_sum is not None and previous_sum is not None and current_sum == previous_sum


def score_op_sum(cumulative_score: float, previous_state: State, current_state: State, output_states: Sequence[State]) -> float:
    """
    Determines the score of the sum operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return: score
    """
    if cumulative_score < 0.0:
        return -1.0
    if validate_op_sum(previous_state, current_state):
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
        name='score_sum',
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
