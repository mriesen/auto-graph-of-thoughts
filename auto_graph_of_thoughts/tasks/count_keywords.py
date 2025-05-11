import re
from collections import Counter
from typing import Mapping, Callable, Set, Sequence

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, relative_complexity, \
    absolute_complexity, ScoreExecOperation, ExecOperation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Evaluator, Task

op_split_4 = PromptOperation(
    name='split_4',
    type=OperationType.GENERATE,
    output_complexity=relative_complexity(1, 4),
    n_inputs=1,
    n_outputs=4,
    prompt=Prompt(
        instruction='Split the following text into 4 paragraphs of approximately same length.'
                    'Output the paragraphs in JSON format.',
        examples=[
            Example(
                input={
                    'text': 'France and Italy are known for their rich cultural heritage and exquisite cuisine, while Japan offers a blend of ancient tradition and cutting-edge technology. Meanwhile, Italy’s scenic countryside, Brazil’s vibrant festivals and Australia’s stunning landscapes attract travelers from around the world.'
                },
                output={
                    'paragraphs': [
                        'France and Italy are known for their rich cultural heritage and exquisite cuisine, ',
                        'while Japan offers a blend of ancient tradition and cutting-edge technology. ',
                        'Meanwhile, Italy’s scenic countryside, Brazil’s vibrant festivals',
                        'and Australia’s stunning landscapes attract travelers from around the world.'
                    ]
                }
            )
        ]
    ),
    transform_before=lambda states: {
        'text': ''
    } if len(states) == 0 else states[0],
    transform_after=lambda state: [{'text': state_paragraph} for state_paragraph in state['paragraphs']]
)

op_merge_4 = PromptOperation(
    name='merge_4',
    type=OperationType.AGGREGATE,
    output_complexity=absolute_complexity(1),
    n_inputs=4,
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
    transform_before=lambda states: {
        'counts': [
            state['counts'] if 'counts' in state
            else None for state in states
            #item for item in [
            #
            #] if item is not None
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


def create_score_op_count(keywords: Set[str]) -> Callable[[float, State, State], float]:
    """
    Creates the score operation for the count operation.
    :param keywords: set of keywords
    :return: score operation for the count operation
    """

    def score_op_count(cumulative_score: float, previous_state: State, current_state: State) -> float:
        """
        Determines the score of the count operation.
        :param cumulative_score: cumulative score
        :param previous_state: previous state
        :param current_state: current state
        :return: score
        """
        if cumulative_score < 0.0:
            return -1.0
        current_count = current_state['count'] if 'count' in current_state else None
        previous_count = (
            _count_keywords(keywords, previous_state['text']) if 'text' in previous_state
            else _count_keywords(keywords, ' '.join(previous_state['paragraphs'])) if 'paragraphs' in previous_state
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

def _create_keep_best_from_10(keywords: Set[str]):
    score_op = create_score_op_count(keywords)
    return ExecOperation(
        name='keep_best_from_10',
        output_complexity=absolute_complexity(1),
        n_inputs=10,
        n_outputs=1,
        type=OperationType.AGGREGATE,
        execute=lambda states: [
            max(states, key=lambda state: score_op(0, state, state), default={})
        ]
    )



def create_count_keywords_task(keywords: Set[str], op_count: PromptOperation) -> Task:
    op_keep_best_from_10 = _create_keep_best_from_10(keywords)
    return Task(
        operations=[op_count, op_split_4, op_merge_4, op_branch_10, op_keep_best_from_10],
        evaluator=Evaluator(
            lambda initial_state, state: 'text' in initial_state
                                         and 'count' in state
                                         and _count_keywords(keywords, initial_state['text']) == state['count']
        )
    )


count_demo_keywords = {
    'France',
    'Italy',
    'Japan',
    'Brazil',
    'Australia',
    'Switzerland'
}
op_count_demo = create_op_count(
    instruction='Count the occurrence of countries in the given text.',
    examples=[
        Example(
            input={
                'text': 'France and Italy are known for their rich cultural heritage and exquisite cuisine, while Japan offers a blend of ancient tradition and cutting-edge technology. Meanwhile, Italy’s scenic countryside, Brazil’s vibrant festivals and Australia’s stunning landscapes attract travelers from around the world.'
            },
            output={
                'count': {
                    'France': 1,
                    'Italy': 2,
                    'Japan': 1,
                    'Brazil': 1,
                    'Australia': 1,
                    'Switzerland': 1
                }
            }
        )
    ],
    keywords=count_demo_keywords
)

count_demo_task = create_count_keywords_task(count_demo_keywords, op_count_demo)
