from typing import Sequence, List, Mapping, Any

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from auto_graph_of_thoughts.tasks.intersect_set import op_intersect

_intersect_set_probabilities: Mapping[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.99,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 0.98,
    11: 0.99,
    12: 0.98,
    13: 1.0,
    14: 0.97,
    15: 0.99,
    16: 0.92,
    17: 0.91,
    18: 0.93,
    19: 0.86,
    20: 0.75,
    21: 0.79,
    22: 0.64,
    23: 0.63,
    24: 0.46,
    25: 0.53,
    26: 0.42,
    27: 0.39,
    28: 0.29,
    29: 0.24,
    30: 0.25,
    31: 0.13,
    32: 0.18
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
        length = min(len(state['set1']), len(state['set2']))
        if length in _intersect_set_probabilities:
            return _intersect_set_probabilities[length]
    return 0.0


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
            )
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
            )
        ])
    return simulated_chat_gpt
