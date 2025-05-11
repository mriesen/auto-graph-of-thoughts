import re
from collections import Counter
from typing import Sequence, Mapping, Dict

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from ..tasks.count_keywords import op_merge_4, op_count_demo, count_demo_keywords, op_split_4

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


def _count_keywords_correctly(prompt: Prompt, state: State) -> State:
    counter: Counter[str] = Counter()
    text = state['text']
    for keyword in count_demo_keywords:
        matches = re.findall(rf'{re.escape(keyword)}', text, flags=re.IGNORECASE)
        count = len(matches)
        if count > 0:
            counter.update({keyword: count})

    return {
        'counts': dict(counter)
    }

def _count_keywords_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'counts': {}
    }


def _get_count_keywords_probability(prompt: Prompt, state: State) -> float:
    if 'text' in state:
        length = len(state['text'].split())

        if length in _count_keywords_probabilities:
            return _count_keywords_probabilities[length]

        complexities = sorted(_count_keywords_probabilities.keys())
        if length > max(complexities):
            return 0.0

        for i in range(1, len(complexities)):
            if length < complexities[i]:
                x0, x1 = complexities[i - 1], complexities[i]
                y0, y1 = _count_keywords_probabilities[x0], _count_keywords_probabilities[x1]

                t = (length - x0) / (x1 - x0)
                return min(1.0, y0 + t * (y1 - y0))

    return 0.0


def _merge_counts(prompt: Prompt, state: State) -> State:
    if 'counts' not in state:
        if 'text' in state:
            return {
                'counts': []
            }

    c: Sequence[Dict[str, int]] = state['counts']
    if len(c) == 0:
        return {
            'counts': []
        }
    if len(c) == 1:
        return {
            'counts': c
        }

    combined = Counter()
    for count_dict in c:
        if isinstance(count_dict, dict):
            combined.update(count_dict)

    return {
        'counts': dict(combined)
    }

def _split_text_4(prompt: Prompt, state: State) -> State:
    if 'text' not in state:
        if 'paragraphs' in state:
            return state
        return {
            'paragraphs': [
                '',
                '',
                '',
                ''
            ]
        }

    t: str = state['text']
    if len(t) == 0:
        return {
            'paragraphs': [
                '',
                '',
                '',
                ''
            ]
        }

    words = t.split()
    part_len = len(words) // 4

    return {
        'paragraphs': [
            ' '.join(words[:part_len]),
            ' '.join(words[part_len:2*part_len]),
            ' '.join(words[2*part_len:3*part_len]),
            ' '.join(words[3*part_len:])
        ]
    }



def create_simulated_realistic_chat_gpt_count_keywords(seed: int) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    :param seed: seed to use for random number generator
    :return: simulated ChatGPT instance for the task sort_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_count_demo.prompt,
                mocked_correct_behavior=_count_keywords_correctly,
                mocked_incorrect_behavior=_count_keywords_incorrectly,
                probability=_get_count_keywords_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split_4.prompt,
                mocked_correct_behavior=_split_text_4,
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge_4.prompt,
                mocked_correct_behavior=_merge_counts,
            ),
        ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_count_keywords(seed: int) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :return: simulated ChatGPT instance for the task sort_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_count_demo.prompt,
                mocked_correct_behavior=_count_keywords_correctly,
                mocked_incorrect_behavior=_count_keywords_incorrectly,
                probability=lambda p, s: 1.0 if _get_count_keywords_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split_4.prompt,
                mocked_correct_behavior=_split_text_4,
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge_4.prompt,
                mocked_correct_behavior=_merge_counts,
            ),
        ])
    return simulated_chat_gpt
