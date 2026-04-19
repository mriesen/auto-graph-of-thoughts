from random import Random
from typing import Tuple, Sequence, Callable

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task


def create_generate_init_state_merge_docs(all_document_groups: Sequence[Sequence[str]]) -> Callable[
    [Random, Sequence[int], Task],
    Tuple[int, State]
]:
    def generate_init_state_merge_docs(rnd: Random, complexities: Sequence[int], task: Task) -> Tuple[int, State]:
        """
        Generates an initial state based on the given random number generator and complexities.
        :param rnd: random number generator
        :param complexities: complexities (number of documents to merge)
        :param task: task
        :return: complexity and initial state
        """
        complexity = rnd.choice(complexities)
        document_group = rnd.choice(all_document_groups)
        init_state: State = {
            'documents': list(document_group[:complexity])
        }
        return complexity, init_state

    return generate_init_state_merge_docs
