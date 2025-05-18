from random import Random
from typing import Tuple, Sequence

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task

from auto_graph_of_thoughts.tasks.count_keywords import count_demo_keywords, count_demo_task, op_count_demo
from auto_graph_of_thoughts.tasks.intersect_set import intersect_set_task
from auto_graph_of_thoughts.tasks.sort_list import sort_list_task
from auto_graph_of_thoughts.tasks.sum_list import sum_list_task


def generate_init_state_sum_list(rnd: Random, complexities: Sequence[int], task: Task) -> Tuple[int, State]:
    """
    Generates an initial state based on the given random number generator and complexities.
    :param task: task
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
