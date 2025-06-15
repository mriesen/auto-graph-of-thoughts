from typing import Set, Optional, Sequence

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, relative_complexity, \
    ScoreExecOperation, ExecOperation, absolute_complexity
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, Evaluator

op_noop = ExecOperation(
    name='noop',
    type=OperationType.GENERATE,
    n_inputs=1,
    n_outputs=1,
    output_complexity=relative_complexity(1),
    execute=lambda states: states
)

op_split = PromptOperation(
        name='split',
        n_outputs=2,
        n_inputs=1,
        type=OperationType.GENERATE,
        output_complexity=relative_complexity(1, 2),
        prompt=Prompt(
                instruction='Split the given sets into two subsets each of equal size. '
                            'Count the number of elements in the set before deciding where to split.'
                            'Only output the sets in JSON format as the examples show.',
                examples=[
                    Example(
                            input={
                                'set1': [8, 2, 0, 1],
                                'set2': [1, 4, 8, 3],
                            },
                            output={
                                'sets1': [
                                    [8, 2],
                                    [0, 1]
                                ],
                                'sets2': [
                                    [1, 4],
                                    [8, 3]
                                ]
                            }
                    )
                ],
        ),
        transform_before=lambda states: {
            'set1': [],
            'set2': [],
        } if len(states) == 0 else states[0],
        transform_after=lambda state: [
            {
                'set1': state['sets1'][i] if 'sets1' in state else [],
                'set2': state['sets2'][i] if 'sets2' in state else [],
            } for i in range(2)
        ]
)

op_merge = PromptOperation(
        name='merge',
        n_outputs=1,
        n_inputs=2,
        type=OperationType.AGGREGATE,
        output_complexity=relative_complexity(2),
        prompt=Prompt(
                instruction='For each top-level sets list, combine the given sets to a single set. '
                            'Only output the set in JSON format as the examples show.',
                examples=[
                    Example(
                            input={
                                'sets1': [
                                    [8, 2],
                                    [0, 1]
                                ],
                                'sets2': [
                                    [1, 4],
                                    [8, 3]
                                ]
                            },
                            output={
                                'set1': [8, 2, 0, 1],
                                'set2': [1, 4, 8, 3],
                            }
                    )
                ]
        ),
        transform_before=lambda states: {
            'sets1': [
                state['set1'] if 'set1' in state else [] for state in states
            ],
            'sets2': [
                state['set2'] if 'set2' in state else [] for state in states
            ]
        }
)


def count_number_of_intersect_errors(
        initial_set1: Set[int], initial_set2: Set[int], current_intersection: Set[int]
) -> int:
    """
    Counts the number of intersect errors for given initial sets and the current intersection.
    :param initial_set1: initial set 1
    :param initial_set2: initial set 2
    :param current_intersection: current intersection
    :return: number of intersect errors
    """
    expected_intersection = initial_set1.intersection(initial_set2)
    return len(expected_intersection.symmetric_difference(current_intersection))


def _get_previous_sets(previous_state: State) -> Optional[Set[int]]:
    """
    Gets the previous sets from the previous state
    :param previous_state: previous state
    :return: previous sets if present
    """
    if (
            'set1' in previous_state
            and 'set2' in previous_state
            and previous_state['set1'] is not None
            and previous_state['set2'] is not None
    ):
        return set(previous_state['set1']).intersection(
                set(previous_state['set2'])
        )
    if 'sets1' in previous_state and 'sets2' in previous_state:
        return {
            element for state_set in previous_state['sets1'] for element in state_set
        }.intersection({
            element for state_set in previous_state['sets2'] for element in state_set
        })
    return None


def score_op_intersect(cumulative_score: float, previous_state: State, current_state: State, output_states: Sequence[State]) -> float:
    """
    Determines the score of the intersection operation.
    :param cumulative_score: cumulative score
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return: score
    """
    if cumulative_score < 0.0:
        return -1.0

    current_intersection = set(current_state['intersection']) if 'intersection' in current_state else None

    previous_sets = _get_previous_sets(previous_state)
    if previous_sets is not None and current_intersection == previous_sets:
        return 1.0
    return -1.0


op_intersect = PromptOperation(
        name='intersect',
        type=OperationType.GENERATE,
        n_inputs=1,
        n_outputs=1,
        output_complexity=relative_complexity(1),
        prompt=Prompt(
                instruction='Find the intersection of two given sets of integers.'
                            'Output only the numbers that are present in both input sets in JSON format.',
                examples=[
                    Example(
                            input={
                                'set1': [1, 5, 8, 2],
                                'set2': [2, 4, 5, 9],
                            },
                            output={
                                'intersection': [2, 5]
                            }
                    )
                ]
        ),
        transform_before=lambda states: {
            'set1': [],
            'set2': []
        } if len(states) == 0 else states[0],
        score_operation=ScoreExecOperation(
                name='score',
                type=OperationType.SCORE,
                score=score_op_intersect,
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
    output_complexity=relative_complexity(1),
    n_inputs=5,
    n_outputs=1,
    type=OperationType.AGGREGATE,
    execute=lambda states: [
        max(states, key=lambda state: score_op_intersect(0, state, state, states), default={})
    ]
)

intersect_set_task = Task(
        operations=[op_intersect, op_branch_5, op_keep_best_from_5],
        evaluator=Evaluator(
                lambda initial_state, state: 'set1' in initial_state and 'set2' in initial_state
                                             and 'intersection' in state
                                             and set(initial_state['set1'])
                                             .intersection(set(initial_state['set2'])) == set(state['intersection'])
        )
)
