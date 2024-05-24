from typing import Sequence, List, Mapping

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior
from ..tasks.sum_list import op_sum, op_split, op_merge

_sum_list_probabilities: Mapping[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 0.98,
    10: 0.88,
    11: 0.89,
    12: 0.75,
    13: 0.74,
    14: 0.58,
    15: 0.5,
    16: 0.34,
    17: 0.3,
    18: 0.21,
    19: 0.1,
    20: 0.09,
    21: 0.12,
    22: 0.05,
    23: 0.09,
    24: 0.04,
    25: 0.04,
    26: 0.04,
    27: 0.02,
    28: 0.0,
    29: 0.0,
    30: 0.01,
    31: 0.0,
    32: 0.05
}


def _sum_list_correctly(prompt: Prompt, state: State) -> State:
    if 'list' not in state:
        return {
            'sum': -1
        }
    l: Sequence[int] = state['list']
    return {
        'sum': sum(l)
    }


def _sum_list_incorrectly(prompt: Prompt, state: State) -> State:
    return {
        'sum': -1
    }


def _get_sum_list_probability(prompt: Prompt, state: State) -> float:
    if 'sum' in state:
        return _sum_list_probabilities[1]
    if 'list' in state:
        length = len(state['list'])
        if length in _sum_list_probabilities:
            return _sum_list_probabilities[length]
    return 0.0


def _merge_lists(prompt: Prompt, state: State) -> State:
    if 'lists' not in state:
        if 'list' in state:
            return state
        if 'sum' in state:
            return {
                'list': [state['sum']]
            }
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


def _split_list(prompt: Prompt, state: State) -> State:
    if 'list' not in state:
        if 'sum' in state:
            return {
                'lists': [
                    [state['sum']],
                    []
                ]
            }
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


def create_simulated_realistic_chat_gpt_sum_list(seed: int) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sum_list.
    :param seed: seed to use for random number generator
    :return: simulated ChatGPT instance for the task sum_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
            seed=seed,
            simulated_behaviors=[
                SimulatedLanguageModelBehavior(
                        prompt=op_sum.prompt,
                        mocked_correct_behavior=_sum_list_correctly,
                        mocked_incorrect_behavior=_sum_list_incorrectly,
                        probability=_get_sum_list_probability
                ),
                SimulatedLanguageModelBehavior(
                        prompt=op_split.prompt,
                        mocked_correct_behavior=_split_list,
                ),
                SimulatedLanguageModelBehavior(
                        prompt=op_merge.prompt,
                        mocked_correct_behavior=_merge_lists,
                ),
            ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_sum_list(seed: int) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task sum_list.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :return: simulated ChatGPT instance for the task sum_list
    """
    simulated_chat_gpt = SimulatedLanguageModel(
            seed=seed,
            simulated_behaviors=[
                SimulatedLanguageModelBehavior(
                        prompt=op_sum.prompt,
                        mocked_correct_behavior=_sum_list_correctly,
                        mocked_incorrect_behavior=_sum_list_incorrectly,
                        probability=lambda p, s: 1.0 if _get_sum_list_probability(p, s) == 1.0 else 0.0
                ),
                SimulatedLanguageModelBehavior(
                        prompt=op_split.prompt,
                        mocked_correct_behavior=_split_list,
                ),
                SimulatedLanguageModelBehavior(
                        prompt=op_merge.prompt,
                        mocked_correct_behavior=_merge_lists,
                ),
            ])
    return simulated_chat_gpt

