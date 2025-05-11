from random import Random
from typing import Tuple, Sequence

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task

from auto_graph_of_thoughts.tasks.count_keywords import count_demo_keywords, count_demo_task, op_count_demo
from auto_graph_of_thoughts.tasks.intersect_set import intersect_set_task
from auto_graph_of_thoughts.tasks.sort_list import sort_list_task
from auto_graph_of_thoughts.tasks.sum_list import sum_list_task


def generate_init_state(rnd: Random, complexities: Sequence[int], task: Task) -> Tuple[int, State]:
    """
    Generates an initial state based on the given random number generator and complexities.
    :param rnd: random number generator
    :param complexities: complexities
    :return:
    """
    complexity = rnd.choice(complexities)
    if task == sum_list_task or task == sort_list_task:
        init_state: State = {
            'list': [
                rnd.randint(0, 9) for _ in range(complexity)
            ]
        }
        return complexity, init_state

    if task == intersect_set_task:
        init_state: State = {
            'set1': list(
                set(
                    rnd.sample(range(0, complexity), complexity)
                )
            ),
            'set2': list(
                set(
                    rnd.sample(range(0, complexity), complexity)
                )
            )
        }
        return complexity, init_state

    if task == count_demo_task:
        text = op_count_demo.prompt.examples[0].input['text'].split()
        init_state: State = {
            'text': ' '.join(text[:complexity]),
        } if len(text) < complexity else {
            'text': ' '.join((text * round(len(text) / complexity))[:complexity]),
        }
        return complexity, init_state

    raise
