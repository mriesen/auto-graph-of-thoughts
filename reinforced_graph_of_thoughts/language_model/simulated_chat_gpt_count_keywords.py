import re
from collections import Counter
from typing import Sequence, Mapping, Dict, Any, Callable

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.operation import PromptOperation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from .simulated_language_model_exception import SimulatedLanguageModelException
from ..tasks.count_keywords import op_merge, op_split

_count_keywords_probabilities: Mapping[int, float] = {
    10: 1.0,
    20: 0.88,
    30: 0.71,
    40: 0.64,
    50: 0.31,
    60: 0.28,
    70: 0.15,
    80: 0.12,
    90: 0.06,
    100: 0.02
}

_split_text_probabilities: Mapping[int, float] = {
    10: 0.62,
    20: 0.41,
    30: 0.75,
    40: 0.76,
    50: 0.74,
    60: 0.76,
    70: 0.82,
    80: 0.73,
    90: 0.85,
    100: 0.8
}


def count_keywords(keywords: Sequence[str], text: str) -> Mapping[str, int]:
    counter: Counter[str] = Counter()
    for keyword in keywords:
        matches = re.findall(rf'{re.escape(keyword)}', text, flags=re.IGNORECASE)
        count = len(matches)
        if count > 0:
            counter.update({keyword: count})
    return dict(counter)


def create_count_keywords_correctly(keywords: Sequence[str]) -> Callable[[Prompt, State], State]:
    def count_keywords_correctly(prompt: Prompt, state: State) -> State:
        if 'text' not in state:
            return {
                'counts': {

                }
            }
        text = state['text']
        counts = count_keywords(keywords, text)
        return {
            'counts': counts
        }

    return count_keywords_correctly


def _count_keywords_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'counts': {}
    }


def _interpolate(length: int, complexities: Sequence[int], probabilities: Mapping[int, float]) -> float:
    for i in range(1, len(complexities)):
        if length < complexities[i]:
            x0, x1 = complexities[i - 1], complexities[i]
            y0, y1 = _count_keywords_probabilities[x0], _count_keywords_probabilities[x1]

            t = (length - x0) / (x1 - x0)
            return min(1.0, y0 + t * (y1 - y0))
    return 0.0


def _get_count_keywords_probability(prompt: Prompt, state: State) -> float:
    if 'text' in state:
        length = len(state['text'].split())

        if length in _count_keywords_probabilities:
            return _count_keywords_probabilities[length]

        complexities = sorted(_count_keywords_probabilities.keys())
        if length > max(complexities):
            return 0.0

        return _interpolate(length, complexities, _count_keywords_probabilities)

    return 0.0


def _get_split_text_probability(prompt: Prompt, state: State) -> float:
    if 'text' in state:
        length = len(state['text'].split())

        if length in _split_text_probabilities:
            return _split_text_probabilities[length]

        complexities = sorted(_split_text_probabilities.keys())
        if length > max(complexities):
            return 0.0

        return _interpolate(length, complexities, _split_text_probabilities)

    return 0.0


def _merge_counts_correctly(prompt: Prompt, state: State) -> State:
    if 'counts' not in state:
        if 'text' in state:
            return {
                'counts': {}
            }

    c: Sequence[Dict[str, int]] = state['counts']
    if len(c) == 0:
        return {
            'counts': {}
        }
    if len(c) == 1:
        return {
            'counts': c[0]
        }

    combined: Counter[str] = Counter()
    for count_dict in c:
        if isinstance(count_dict, dict):
            combined.update(count_dict)

    return {
        'counts': dict(combined)
    }


def _merge_counts_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'counts': {}
    }


def split_text(text: str) -> Sequence[str]:
    words = text.split()
    part_len = len(words) // 2
    return [
        ' '.join(words[:part_len]),
        ' '.join(words[part_len:])
    ]


def _split_text_correctly(prompt: Prompt, state: State) -> State:
    if 'text' not in state:
        if 'texts' in state:
            return state
        return {
            'texts': [
                '',
                ''
            ]
        }

    t: str = state['text']
    if len(t) == 0:
        return {
            'texts': [
                '',
                ''
            ]
        }

    split_texts = split_text(t)

    return {
        'texts': split_texts
    }


def _split_text_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'texts': ['', '']
    }


def create_simulated_realistic_chat_gpt_count_keywords(seed: int,
                                                       extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments
    :return: simulated ChatGPT instance for the task sort_list
    """
    if 'keywords' not in extra_args:
        raise SimulatedLanguageModelException('extra_args must contain keywords')
    keywords = extra_args['keywords']

    if 'op_count' not in extra_args or not isinstance(extra_args['op_count'], PromptOperation):
        raise SimulatedLanguageModelException('extra_args must contain op_count PromptOperation')
    op_count: PromptOperation = extra_args['op_count']

    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_count.prompt,
                mocked_correct_behavior=create_count_keywords_correctly(keywords),
                mocked_incorrect_behavior=_count_keywords_incorrectly,
                probability=_get_count_keywords_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_text_correctly,
                mocked_incorrect_behavior=_split_text_incorrectly,
                probability=_get_split_text_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_counts_correctly,
                mocked_incorrect_behavior=_merge_counts_incorrectly
            ),
        ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_count_keywords(seed: int,
                                                           extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments
    :return: simulated ChatGPT instance for the task sort_list
    """
    if 'keywords' not in extra_args:
        raise SimulatedLanguageModelException('extra_args must contain keywords')
    keywords = extra_args['keywords']

    if 'op_count' not in extra_args or not isinstance(extra_args['op_count'], PromptOperation):
        raise SimulatedLanguageModelException('extra_args must contain op_count PromptOperation')
    op_count: PromptOperation = extra_args['op_count']

    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_count.prompt,
                mocked_correct_behavior=create_count_keywords_correctly(keywords),
                mocked_incorrect_behavior=_count_keywords_incorrectly,
                probability=lambda p, s: 1.0 if _get_count_keywords_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_text_correctly,
                mocked_incorrect_behavior=_split_text_incorrectly,
                probability=lambda p, s: 1.0 if _get_split_text_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_counts_correctly,
                mocked_incorrect_behavior=_merge_counts_incorrectly
            ),
        ])
    return simulated_chat_gpt
