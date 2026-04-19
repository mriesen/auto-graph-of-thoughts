import re
from collections import Counter
from typing import Mapping, Callable, Set, Sequence

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, relative_complexity, \
    absolute_complexity, ScoreExecOperation, ExecOperation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Evaluator, Task


def validate_op_split(previous_state: State, current_state: State, output_states: Sequence[State]) -> bool:
    """
    Checks whether the split operation was performed correctly.
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return:
    """
    min_tolerated_text_ratio = 0.25
    if 'text' in previous_state and len(output_states) == 2 and all(
            ['text' in output_state for output_state in output_states]
    ):
        previous_text: str = previous_state['text']
        output_texts: Sequence[str] = [output_state['text'] for output_state in output_states]
        output_text_1, output_text_2 = output_texts[0], output_texts[1]
        text_concatenated = ''.join([output_text_1, output_text_2]).replace(' ', '')
        text_concatenated_reversed = ''.join([output_text_2, output_text_1]).replace(' ', '')
        lengths = [len(t) for t in output_texts]
        min_len, max_len = min(lengths), max(lengths)
        text_ratio = min_len / max_len if max_len > 0 else 0.0
        text_equals = text_concatenated == previous_text.replace(' ', '')
        text_equals_reversed = text_concatenated_reversed == previous_text.replace(' ', '')
        return (text_equals or text_equals_reversed) and text_ratio >= min_tolerated_text_ratio
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
    type=OperationType.GENERATE,
    output_complexity=relative_complexity(1, 2),
    n_inputs=1,
    n_outputs=2,
    prompt=Prompt(
        instruction='Split the given text into two substrings of equal length (number of words).'
                    'Make sure the number of words is equal for both texts, split exactly in the half of the given text.'
                    'Output the texts in JSON format.',
        examples=[
            Example(
                input={
                    'text': 'France and Italy are known for their rich cultural heritage and exquisite cuisine, while Japan offers a blend of ancient tradition and cutting-edge technology. Meanwhile, Italy’s scenic countryside, Brazil’s vibrant festivals and Australia’s stunning landscapes attract travelers from around the world.'
                },
                output={
                    'texts': [
                        'France and Italy are known for their rich cultural heritage and exquisite cuisine, while Japan offers a blend of ancient tradition and cutting-edge',
                        'technology. Meanwhile, Italy’s scenic countryside, Brazil’s vibrant festivals and Australia’s stunning landscapes attract travelers from around the world.'
                    ]
                }
            )
        ]
    ),
    score_operation=ScoreExecOperation(
        name='score_split',
        type=OperationType.SCORE,
        score=score_op_split,
        n_inputs=1,
        n_outputs=1
    ),
    transform_before=lambda states: {
        'text': ''
    } if len(states) == 0 else states[0],
    transform_after=lambda state: [{'text': state_paragraph} for state_paragraph in state['texts']]
)


def validate_op_merge(previous_state: State, current_state: State, output_states: Sequence[State]) -> bool:
    """
    Checks whether the merge operation was performed correctly.
    :param previous_state: previous state
    :param current_state: current state
    :param output_states: output states
    :return:
    """
    if 'counts' in previous_state and 'counts' in current_state:
        combined: Counter[str] = Counter()
        for count_dict in previous_state['counts']:
            if isinstance(count_dict, dict):
                combined.update(count_dict)
        return dict(combined) == dict(current_state['counts'])
    return False


def score_op_merge(cumulative_score: float, previous_state: State, current_state: State,
                   output_states: Sequence[State]) -> float:
    """
    Determines the score of the merge operation.
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
    type=OperationType.AGGREGATE,
    output_complexity=absolute_complexity(1),
    n_inputs=2,
    n_outputs=1,
    prompt=Prompt(
        instruction='Combine the given dictionaries into a single one. Sum the values to aggregate the number of country occurrences.'
                    'Output the country occurrences in JSON format.',
        examples=[
            Example(
                input={
                    'counts': [
                        {
                            'France': 1,
                            'Italy': 1,
                            'Japan': 1,
                            'Brazil': 1,
                            'Australia': 1
                        },
                        {
                            'Italy': 1,
                            'Switzerland': 1
                        },
                    ]
                },
                output={
                    'counts': {
                        'France': 1,
                        'Italy': 2,
                        'Japan': 1,
                        'Brazil': 1,
                        'Australia': 1,
                        'Switzerland': 1
                    }
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
        'counts': [
            state['counts'] if 'counts' in state
            else None for state in states
            # item for item in [
            #
            # ] if item is not None
        ]
    }
)

op_branch_10 = ExecOperation(
    name='branch_10',
    output_complexity=relative_complexity(1),
    n_inputs=1,
    n_outputs=10,
    type=OperationType.GENERATE,
    execute=lambda states: [
        states[0] for _ in range(10)
    ]
)


def _count_keywords(keywords: Set[str], text: str) -> Mapping[str, int]:
    """
    Counts the number of keywords in a text.
    :param keywords: set of keywords
    :param text: text to search for keywords in
    :return: count result
    """
    counter: Counter[str] = Counter()
    for keyword in keywords:
        matches = re.findall(rf'{re.escape(keyword)}', text, flags=re.IGNORECASE)
        count = len(matches)
        if count > 0:
            counter.update({keyword: count})

    return counter


def count_number_of_count_errors(keywords: Set[str], text: str, current_count: Mapping[str, int]) -> int:
    """
    Counts the number of count errors for a given text and count.
    :param keywords: set of keywords
    :param text: text to count keywords of
    :param current_count: current count
    :return: number of errors
    """
    actual_count = _count_keywords(keywords, text)
    errors = 0
    for key, value in actual_count.items():
        current_value = current_count[key] if key in current_count else 0
        errors += abs(value - current_value)

    return errors


def create_score_op_count(keywords: Set[str]) -> Callable[[float, State, State, Sequence[State]], float]:
    """
    Creates the score operation for the count operation.
    :param keywords: set of keywords
    :return: score operation for the count operation
    """

    def score_op_count(cumulative_score: float, previous_state: State, current_state: State,
                       output_states: Sequence[State]) -> float:
        """
        Determines the score of the count operation.
        :param cumulative_score: cumulative score
        :param previous_state: previous state
        :param current_state: current state
        :param output_states: output states
        :return: score
        """
        if cumulative_score < 0.0:
            return -1.0
        current_count = current_state['counts'] if 'counts' in current_state else None
        previous_count = (
            _count_keywords(keywords, previous_state['text']) if 'text' in previous_state
            else _count_keywords(keywords, ' '.join(previous_state['texts'])) if 'texts' in previous_state
            else None
        )
        if current_count is not None and previous_count is not None and current_count == previous_count:
            return 1.0
        return -1.0

    return score_op_count


def create_op_count(instruction: str, examples: Sequence[Example], keywords: Set[str]) -> PromptOperation:
    """
    Creates the count operation based on an instruction and the text keyword lookup table.
    :param examples: examples in the prompt
    :param instruction: instruction (without the instruction about the JSON format output)
    :param keywords: set of keywords
    :return: count operation
    """
    return PromptOperation(
        name='count',
        type=OperationType.GENERATE,
        n_inputs=1,
        n_outputs=1,
        output_complexity=relative_complexity(1),
        prompt=Prompt(
            instruction=instruction +
                        'Only output the count in JSON format.',
            examples=examples
        ),
        score_operation=ScoreExecOperation(
            name='score',
            type=OperationType.SCORE,
            score=create_score_op_count(keywords),
            n_inputs=1,
            n_outputs=1
        )
    )


def _create_keep_best_from_10(keywords: Set[str]) -> ExecOperation:
    score_op = create_score_op_count(keywords)
    return ExecOperation(
        name='keep_best_from_10',
        output_complexity=absolute_complexity(1),
        n_inputs=10,
        n_outputs=1,
        type=OperationType.AGGREGATE,
        execute=lambda states: [
            max(states, key=lambda state: score_op(0, state, state, states), default={})
        ]
    )


def create_count_keywords_task(keywords: Set[str], op_count: PromptOperation) -> Task:
    op_keep_best_from_10 = _create_keep_best_from_10(keywords)
    return Task(
        operations=[op_count, op_split, op_merge, op_branch_10, op_keep_best_from_10],
        evaluator=Evaluator(
            lambda initial_state, state: 'text' in initial_state
                                         and 'counts' in state
                                         and _count_keywords(keywords, initial_state['text']) == state['counts']
        )
    )

