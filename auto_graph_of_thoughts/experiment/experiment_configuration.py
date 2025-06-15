from dataclasses import dataclass, field
from random import Random
from typing import Sequence, Set, Callable, Tuple, Mapping, Any, Optional

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task

from .language_model_simulation_type import LanguageModelSimulationType
from ..env import GraphStepRewardVersion
from ..obs import ObservationComponent
from auto_graph_of_thoughts.experiment.experiment_task_type import ExperimentTaskType


@dataclass(frozen=True)
class ExperimentConfiguration:
    """
    Represents the configuration of an experiment.
    """
    seed: int
    """The seed for the PRNG"""

    task: Task
    """The task to use"""

    reward_version: GraphStepRewardVersion
    """The reward version to use"""

    max_steps: int
    """The maximum number of steps per episode"""

    observation_filter: Set[ObservationComponent]
    """The observation space filter"""

    max_depth: int
    """The maximum depth of the graph of operations"""

    max_breadth: int
    """The maximum breadth of the graph of operations"""

    divergence_cutoff_factor: float
    """The divergence cutoff factor of the graph of operations"""

    train_complexities: Sequence[int]
    """The complexities to use for training"""

    eval_complexities: Sequence[int]
    """The complexities to use for evaluation"""

    max_complexity: int
    """The maximum complexity of a state"""

    max_operations: int
    """The maximum number of operations"""

    lm_simulation_type: LanguageModelSimulationType
    """The type of language model simulation"""

    generate_init_state: Callable[[Random, Sequence[int], Task], Tuple[int, State]]
    """The initial state generator returning a tuple of complexity and initial state"""

    task_type: Optional[ExperimentTaskType] = field(default=None)
    """The task type to use, can be derived by the task for most cases"""

    extra_args: Mapping[str, Any] = field(default_factory=dict)
    """The extra arguments to pass to the language model simulation factory"""
