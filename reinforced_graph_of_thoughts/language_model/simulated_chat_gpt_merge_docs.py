from typing import Mapping, Any, List

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import SimulatedLanguageModel, SimulatedLanguageModelBehavior

from ..tasks.merge_docs import op_merge, op_improve

_merge_docs_probabilities: Mapping[int, float] = {
     1: 0.96,
     2: 0.68,
     3: 0.43,
     4: 0.31
}

_improve_docs_probabilities: Mapping[int, float] = {
    1: 0.96
}


def _merge_docs_correctly(prompt: Prompt, state: State) -> State:
    documents: List[str] = state.get('documents', [])
    if not documents:
        return {'merged': ''}
    return {'merged': ' '.join(documents)}


def _merge_docs_incorrectly(prompt: Prompt, state: State) -> State:
    return {'merged': ''}


def _get_merge_docs_probability(prompt: Prompt, state: State) -> float:
    documents: List[str] = state.get('documents', [])
    cardinality = len(documents)
    if cardinality in _merge_docs_probabilities:
        return _merge_docs_probabilities[cardinality]
    return 0.0


def _improve_docs_correctly(prompt: Prompt, state: State) -> State:
    merged: str = state.get('merged', '')
    if merged:
        return {'merged': merged}
    documents: List[str] = state.get('documents', [])
    return {'merged': ' '.join(documents)}


def _improve_docs_incorrectly(prompt: Prompt, state: State) -> State:
    return {'merged': ''}


def _get_improve_docs_probability(prompt: Prompt, state: State) -> float:
    documents: List[str] = state.get('documents', [])
    merged: List[str] = [state.get('merged', '')]
    cardinality = max(len(merged), len(documents))
    if cardinality in _improve_docs_probabilities:
        return _improve_docs_probabilities[cardinality]
    return 0.0


def create_simulated_realistic_chat_gpt_merge_docs(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task merge_docs.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task merge_docs
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_docs_correctly,
                mocked_incorrect_behavior=_merge_docs_incorrectly,
                probability=_get_merge_docs_probability
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_improve.prompt,
                mocked_correct_behavior=_improve_docs_correctly,
                mocked_incorrect_behavior=_improve_docs_incorrectly,
                probability=_get_improve_docs_probability
            ),
        ])
    return simulated_chat_gpt


def create_simulated_deterministic_chat_gpt_merge_docs(seed: int, extra_args: Mapping[str, Any]) -> SimulatedLanguageModel:
    """
    Creates a simulated ChatGPT instance for the task merge_docs.
    The probabilities are either 1.0 or 0.0.
    :param seed: seed to use for random number generator
    :param extra_args: extra arguments, ignored for this function
    :return: simulated ChatGPT instance for the task merge_docs
    """
    simulated_chat_gpt = SimulatedLanguageModel(
        seed=seed,
        simulated_behaviors=[
            SimulatedLanguageModelBehavior(
                prompt=op_merge.prompt,
                mocked_correct_behavior=_merge_docs_correctly,
                mocked_incorrect_behavior=_merge_docs_incorrectly,
                probability=lambda p, s: 1.0 if _get_merge_docs_probability(p, s) == 1.0 else 0.0
            ),
            SimulatedLanguageModelBehavior(
                prompt=op_improve.prompt,
                mocked_correct_behavior=_improve_docs_correctly,
                mocked_incorrect_behavior=_improve_docs_incorrectly,
                probability=lambda p, s: 1.0 if _get_improve_docs_probability(p, s) == 1.0 else 0.0
            ),
        ])
    return simulated_chat_gpt