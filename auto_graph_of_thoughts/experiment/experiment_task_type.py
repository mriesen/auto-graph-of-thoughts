from __future__ import annotations

from enum import Enum

from pure_graph_of_thoughts.api.task import Task

from auto_graph_of_thoughts.experiment.experiment_task_type_lookup_exception import ExperimentTaskTypeLookupException
from auto_graph_of_thoughts.tasks.intersect_set import intersect_set_task
from auto_graph_of_thoughts.tasks.sort_list import sort_list_task
from auto_graph_of_thoughts.tasks.sum_list import sum_list_task


class ExperimentTaskType(Enum):
    """
    Represents the type of task of an experiment.
    """
    SUM_LIST = 'sum_list'
    SORT_LIST = 'sort_list'
    INTERSECT_SET = 'intersect_set'
    COUNT_KEYWORDS = 'count_keywords'

    @classmethod
    def from_task(cls, task: Task) -> ExperimentTaskType:
        if task == sum_list_task:
            return cls.SUM_LIST
        if task == sort_list_task:
            return cls.SORT_LIST
        if task == intersect_set_task:
            return cls.INTERSECT_SET

        raise ExperimentTaskTypeLookupException(f"Unknown task type: {task}")
