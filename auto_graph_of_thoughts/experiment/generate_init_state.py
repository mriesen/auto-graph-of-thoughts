from random import Random
from typing import Tuple, Sequence

from pure_graph_of_thoughts.api.state import State


def generate_init_state(rnd: Random, complexities: Sequence[int]) -> Tuple[int, State]:
    """
    Generates an initial state based on the given random number generator and complexities.
    :param rnd: random number generator
    :param complexities: complexities
    :return:
    """
    complexity = rnd.choice(complexities)
    init_state: State = {
        'list': [
            rnd.randint(0, 9) for _ in range(complexity)
        ]

    }
    return complexity, init_state
