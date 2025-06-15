from typing import Sequence, List, Mapping, Any

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from auto_graph_of_thoughts.tasks.intersect_set import op_merge, op_split, op_intersect

_intersect_set_probabilities: Mapping[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.98,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 0.98,
    11: 0.99,
    12: 0.99,
    13: 1.0,
    14: 0.97,
    15: 0.99,
    16: 0.91,
    17: 0.9,
    18: 0.89,
    19: 0.88,
    20: 0.73,
    21: 0.82,
    22: 0.63,
    23: 0.55,
    24: 0.46,
    25: 0.51,
    26: 0.45,
    27: 0.36,
    28: 0.26,
    29: 0.26,
    30: 0.21,
    31: 0.17,
    32: 0.16
}


def _intersect_set_correctly(prompt: Prompt, state: State) -> State:
    if 'set1' not in state or 'set2' not in state:
        return {
            'intersection': []
        }
    return {
        'intersection': list(set(state['set1']).intersection(set(state['set2'])))
    }


def _intersect_set_incorrectly(prompt: Prompt, state: State) -> State:
    if 'set1' not in state or 'set2' not in state:
        return {
            'intersection': []
        }
    return {
        'intersection': list(set(state['set1']).difference(set(state['set2'])))
    }


def _get_intersect_set_probability(prompt: Prompt, state: State) -> float:
    if 'set1' in state and 'set2' in state:
        length = max(len(state['set1']), len(state['set2']))
        if length in _intersect_set_probabilities:
            return _intersect_set_probabilities[length]
    return 0.0

def _merge_sets(prompt: Prompt, state: State) -> State:
    if 'sets1' not in state or 'sets2' not in state:
        if 'set1' in state and 'set2' in state:
            return state
    sets1: Sequence[List[int]] = state['sets1']
    sets2: Sequence[List[int]] = state['sets2']
    if len(sets1) > 2 and len(sets2) > 2:
        return {
            'set1': [],
            'set2': []
        }
    return {
        'set1': list(set(list(sets1[0]) + list(sets1[1]))),
        'set2': list(set(list(sets2[0]) + list(sets2[1]))),
    }


def _split_sets(prompt: Prompt, state: State) -> State:
    if 'set1' not in state or 'set2' not in state:
        if 'sets1' in state and 'sets2' in state:
            return state
    if 'intersection' in state:
        return state
    set1: Sequence[int] = list(state['set1'])
    set2: Sequence[int] = list(state['set2'])
    if len(set1) == 0:
        return {
            'sets1': [
                [],
                []
            ],
            'sets2': [
                [],
                []
            ]
        }
    return {
        'sets1': [
            set(set1[:len(set1) // 2]),
            set(set1[len(set1) // 2:])
        ],
        'sets2': [
            set(set2[:len(set2) // 2]),
            set(set2[len(set2) // 2:])
        ]
    }


def create_simulated_realistic_chat_gpt_intersect_set(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task intersect_set.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task intersect_set
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_intersect.prompt,
                mocked_correct_behavior=_intersect_set_correctly,
                mocked_incorrect_behavior=_intersect_set_incorrectly,
                probability=_get_intersect_set_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_sets,
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_sets,
            ),
        ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_intersect_set(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task intersect_set.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task intersect_set
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_intersect.prompt,
                mocked_correct_behavior=_intersect_set_correctly,
                mocked_incorrect_behavior=_intersect_set_incorrectly,
                probability=lambda p, s: 1.0 if _get_intersect_set_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_split.prompt,
                mocked_correct_behavior=_split_sets,
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_sets,
            ),
        ])
    return simulated_chat_gpt
