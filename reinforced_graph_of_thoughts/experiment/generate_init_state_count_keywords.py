from random import Random
from typing import Tuple, Sequence, Callable

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task


def create_generate_init_state_count_keywords(all_texts: Sequence[str]) -> Callable[
    [Random, Sequence[int], Task],
    Tuple[int, State]
]:
    def generate_init_state_count_keywords(rnd: Random, complexities: Sequence[int], task: Task) -> Tuple[int, State]:
        """
        Generates an initial state based on the given random number generator and complexities.
        :param rnd: random number generator
        :param complexities: complexities
        :param task: task
        :return:
        """
        complexity = rnd.choice(complexities)

        text = rnd.choice(all_texts)
        words = text.split()
        init_state: State = {
            'text': ' '.join(words[:complexity]),
        } if len(words) < complexity else {
            'text': ' '.join((words * round(len(words) / complexity))[:complexity]),
        }
        return complexity, init_state

    return generate_init_state_count_keywords
