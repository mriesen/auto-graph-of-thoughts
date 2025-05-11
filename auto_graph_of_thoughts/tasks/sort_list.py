from typing import Sequence, Optional

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, relative_complexity, \
    ScoreExecOperation, ExecOperation, absolute_complexity
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
                instruction='Combine the given sorted lists to a single sorted list.'
                            'Apply a merge sort to sort the final list.'
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
                    else None for state in states
                ] if item is not None
            ]
        }
)


def count_number_of_sort_errors(sorted_list: Sequence[int], current_list: Sequence[int]) -> Optional[int]:
    """
    Counts the number or sort errors in the given current list, based on a given sorted list.
    :param sorted_list: sorted list (expected)
    :param current_list: current list (actual)
    :return: number of errors
    """
    if len(sorted_list) != len(current_list):
        return None

    error_count = 0
    for i in range(len(sorted_list)):
        if sorted_list[i] != current_list[i]:
            error_count += 1

    return error_count


def score_op_sort(cumulative_score: float, previous_state: State, current_state: State) -> float:
    """
    Determines the score of the sort operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :return: score
    """

    if cumulative_score < 0.0:
        return -1.0
    current_list = current_state['list'] if 'list' in current_state else None
    sorted_list = (
        sorted(previous_state['list']) if 'list' in previous_state
        else sorted(
                list_item for state_list in previous_state['lists'] for list_item in state_list
        ) if 'lists' in previous_state
        else None
    )
    if current_list is not None and sorted_list is not None:
        num_errors = count_number_of_sort_errors(sorted_list, current_list)
        if num_errors is not None and num_errors == 0:
            return 1.0
    return -1.0

op_sort = PromptOperation(
        name='sort',
        type=OperationType.GENERATE,
        n_inputs=1,
        n_outputs=1,
        output_complexity=relative_complexity(1),
        prompt=Prompt(
                instruction='Sort the given list of single-digit integers in ascending order. Output the sorted list in JSON format.',
                examples=[
                    Example(
                            input={
                                'list': [4, 9, 8, 4, 9, 1, 5, 6, 2, 8, 9, 9, 9, 2, 1, 5]
                            },
                            output={
                                'list': [1, 1, 2, 2, 4, 4, 5, 5, 6, 8, 8, 9, 9, 9, 9, 9]
                            }
                    )
                ]
        ),
        transform_before=lambda states: {
            'list': []
        } if len(states) == 0 else states[0],
        score_operation=ScoreExecOperation(
                name='score',
                type=OperationType.SCORE,
                score=score_op_sort,
                n_inputs=1,
                n_outputs=1
        )
)


op_branch_5 = ExecOperation(
    name='branch_5',
    output_complexity=relative_complexity(1),
    n_inputs=1,
    n_outputs=5,
    type=OperationType.GENERATE,
    execute=lambda states: [
        states[0] for _ in range(5)
    ]
)

op_keep_best_from_5 = ExecOperation(
    name='keep_best_from_5',
    output_complexity=absolute_complexity(1),
    n_inputs=5,
    n_outputs=1,
    type=OperationType.AGGREGATE,
    execute=lambda states: [
        max(states, key=lambda state: score_op_sort(0, state, state), default={})
    ]
)

sort_list_task = Task(
        operations=[op_sort, op_split, op_merge, op_branch_5, op_keep_best_from_5],
        evaluator=Evaluator(
                lambda initial_state, state: 'list' in initial_state
                                             and 'list' in state
                                             and sorted(initial_state['list']) == state['list']
        )
)
