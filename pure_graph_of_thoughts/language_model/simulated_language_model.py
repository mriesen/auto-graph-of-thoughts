from dataclasses import dataclass, field
from random import Random
from typing import Callable, Sequence

from pure_graph_of_thoughts.api.language_model import Prompt
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.language_model import MockLanguageModel


def _identity_behavior(prompt: Prompt, state: State) -> State:
    return state


@dataclass(frozen=True)
class SimulatedLanguageModelBehavior:
    """
    Represents a simulated language model behavior.
    """

    prompt: Prompt
    """The prompt of this behavior"""

    mocked_correct_behavior: Callable[[Prompt, State], State]
    """The correct mocked behavior"""

    mocked_incorrect_behavior: Callable[[Prompt, State], State] = field(
            default_factory=lambda: lambda prompt, state: state
    )
    """The incorrect mocked behavior"""

    probability: Callable[[Prompt, State], float] = field(default_factory=lambda: lambda prompt, state: 1.0)
    """The probability of of a correct behavior"""


class SimulatedLanguageModel(MockLanguageModel):
    """
    Represents a simulated language model.
    The simulation is implemented by using a probability distribution to determine the model's performance.
    """

    def __init__(self, seed: int, simulated_behaviors: Sequence[SimulatedLanguageModelBehavior]):
        self._seed = seed
        rnd = Random(self._seed)
        mocked_behaviors = {
            simulated_behavior.prompt: self._create_mocked_behavior(rnd, simulated_behavior)
            for simulated_behavior in simulated_behaviors
        }
        super().__init__(mocked_behaviors)

    @staticmethod
    def _create_mocked_behavior(
            rnd: Random, simulated_behavior: SimulatedLanguageModelBehavior
    ) -> Callable[[Prompt, State], State]:
        def simulate_behavior(prompt: Prompt, state: State) -> State:
            probability = simulated_behavior.probability(prompt, state)
            if rnd.random() < probability:
                return simulated_behavior.mocked_correct_behavior(prompt, state)
            return simulated_behavior.mocked_incorrect_behavior(prompt, state)

        return simulate_behavior
