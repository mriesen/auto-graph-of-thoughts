from typing import Sequence, List, Mapping, Any

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from ..tasks.sort_list import op_sort, op_split, op_merge

_sort_list_probabilities: Mapping[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 0.99,
    7: 1.0,
    8: 0.97,
    9: 0.97,
    10: 0.97,
    11: 0.94,
    12: 0.78,
    13: 0.71,
    14: 0.68,
    15: 0.74,
    16: 0.91,
    17: 0.83,
    18: 0.84,
    19: 0.74,
    20: 0.65,
    21: 0.54,
    22: 0.5,
    23: 0.48,
    24: 0.51,
    25: 0.38,
    26: 0.4,
    27: 0.29,
    28: 0.25,
    29: 0.11,
    30: 0.17,
    31: 0.12,
    32: 0.14
}

_split_list_probabilities: Mapping[int, float] = {
    # filled in values, not measured
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    # measured values
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 1.0,
    12: 0.98,
    13: 0.99,
    14: 0.9,
    15: 0.87,
    16: 1.0,
    17: 0.97,
    18: 0.97,
    19: 0.99,
    20: 0.99,
    21: 1.0,
    22: 0.96,
    23: 0.93,
    24: 0.97,
    25: 0.83,
    26: 0.83,
    27: 0.79,
    28: 0.53,
    29: 0.75,
    30: 0.65,
    31: 0.53,
    32: 0.77,
    33: 0.79,
    34: 0.75,
    35: 0.8,
    36: 0.94,
    37: 0.93,
    38: 0.72,
    39: 0.85,
    40: 0.81,
    41: 0.88,
    42: 0.92,
    43: 0.67,
    44: 0.94,
    45: 0.83,
    46: 0.83,
    47: 0.68,
    48: 0.84,
    49: 0.98,
    50: 0.92,
    51: 0.8,
    52: 0.91,
    53: 0.97,
    54: 0.55,
    55: 0.74,
    56: 0.87,
    57: 0.9,
    58: 0.86,
    59: 0.83,
    60: 0.5,
    61: 0.87,
    62: 0.78,
    63: 0.95,
    64: 0.75
}

_merge_list_probabilities: Mapping[int, float] = {
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 1.0,
    12: 1.0,
    13: 1.0,
    14: 1.0,
    15: 0.99,
    16: 1.0,
    17: 1.0,
    18: 1.0,
    19: 1.0,
    20: 1.0,
    21: 1.0,
    22: 0.98,
    23: 1.0,
    24: 0.99,
    25: 0.97,
    26: 0.98,
    27: 1.0,
    28: 1.0,
    29: 0.98,
    30: 0.98,
    31: 0.99,
    32: 0.99
}


def _sort_list_correctly(prompt: Prompt, state: State) -> State:
    if 'list' not in state:
        return {
            'list': []
        }
    return {
        'list': sorted(state['list'])
    }


def _sort_list_incorrectly(prompt: Prompt, state: State) -> State:
    return state


def _get_sort_list_probability(prompt: Prompt, state: State) -> float:
    if 'list' in state:
        length = len(state['list'])
        if length in _sort_list_probabilities:
            return _sort_list_probabilities[length]
    return 0.0


def _get_split_list_probability(prompt: Prompt, state: State) -> float:
    if 'sum' in state:
        return _split_list_probabilities[1]
    if 'list' in state:
        length = len(state['list'])
        if length in _split_list_probabilities:
            return _split_list_probabilities[length]
    return 0.0


def _get_merge_list_probability(prompt: Prompt, state: State) -> float:
    length = len(state['list']) if 'list' in state else len(state['lists'][0]) if ('lists' in state and len(state['lists']) == 2) else -1
    if length in _merge_list_probabilities:
        return _merge_list_probabilities[length]
    return 0.0


def _merge_lists_correctly(prompt: Prompt, state: State) -> State:
    if 'lists' not in state:
        if 'list' in state:
            return state
    l: Sequence[List[int]] = state['lists']
    if len(l) == 0:
        return {
            'list': []
        }
    if len(l) == 1:
        return {
            'list': l[0]
        }
    l1 = l[0]
    l2 = l[1]
    if isinstance(l1, list) and isinstance(l2, list):
        return {
            'list': sorted(l[0] + l[1])
        }
    raise


def _merge_lists_incorrectly(prompt: Prompt, state: State) -> State:
    if 'lists' not in state:
        if 'list' in state:
            return state
    l: Sequence[List[int]] = state['lists']
    if len(l) == 0:
        return {
            'list': []
        }
    if len(l) == 1:
        return {
            'list': l[0]
        }
    l1 = l[0]
    l2 = l[1]
    if isinstance(l1, list) and isinstance(l2, list):
        return {
            'list': l[0] + l[1]
        }
    raise


def _split_list_correctly(prompt: Prompt, state: State) -> State:
    if 'list' not in state:
        if 'lists' in state:
            return state
    l: Sequence[int] = state['list']
    if len(l) == 0:
        return {
            'lists': [
                [],
                []
            ]
        }
    return {
        'lists': [
            l[:len(l) // 2],
            l[len(l) // 2:]
        ]
    }


def _split_list_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'lists': []
    }


def create_simulated_realistic_chat_gpt_sort_list(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task sort_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_sort.prompt,
                mocked_correct_behavior=_sort_list_correctly,
                mocked_incorrect_behavior=_sort_list_incorrectly,
                probability=_get_sort_list_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_list_correctly,
                mocked_incorrect_behavior=_split_list_incorrectly,
                probability=_get_split_list_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_lists_correctly,
                mocked_incorrect_behavior=_merge_lists_incorrectly,
                probability=_get_merge_list_probability
            ),
        ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_sort_list(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sort_list.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task sort_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_sort.prompt,
                mocked_correct_behavior=_sort_list_correctly,
                mocked_incorrect_behavior=_sort_list_incorrectly,
                probability=lambda p, s: 1.0 if _get_sort_list_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_list_correctly,
                mocked_incorrect_behavior=_split_list_incorrectly,
                probability=lambda p, s: 1.0 if _get_split_list_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_lists_correctly,
                mocked_incorrect_behavior=_sort_list_incorrectly,
                probability=lambda p, s: 1.0 if _get_merge_list_probability(p, s) == 1.0 else 0.0
            ),
        ])
    return simulated_chat_gpt
