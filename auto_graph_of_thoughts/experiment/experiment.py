from random import Random
from typing import Sequence, Tuple, Optional

from .experiment_configuration import ExperimentConfiguration
from ..controller import ContinuousGraphController
from ..env import GraphOfThoughtsEnv
from ..env.wrapper import DictObsFilterWrapper
from auto_graph_of_thoughts.experiment.experiment_task_type import ExperimentTaskType


class Experiment:
    """
    Represents an experiment.
    """

    _config: ExperimentConfiguration

    @property
    def config(self) -> ExperimentConfiguration:
        """The experiment configuration"""
        return self._config

    def __init__(self, config: ExperimentConfiguration) -> None:
        self._config = config

    def create_unwrapped_train_env(self) -> GraphOfThoughtsEnv:
        """
        Creates an unwrapped training environment.
        :return: unwrapped training environment
        """
        controller = self._create_controller(self._config, self._config.train_complexities)
        return self._create_env(self._config, controller)

    def create_filtered_train_env(self) -> DictObsFilterWrapper:
        """
        Creates a filtered train environment.
        :return: filtered train environment
        """
        controller = self._create_controller(self._config, self._config.train_complexities)
        env = self._create_env(self._config, controller)
        return self._create_filtered_env(self._config, env)

    def created_eval_env_tuple(
            self, eval_complexities: Optional[Sequence[int]] = None
    ) -> Tuple[GraphOfThoughtsEnv, DictObsFilterWrapper]:
        """
        Creates a filtered evaluation environment.
        :param eval_complexities: complexities to evaluate
        :return: tuple of unwrapped environment and filtered environment
        """
        if eval_complexities is None:
            eval_complexities = self._config.eval_complexities
        controller = self._create_controller(self._config, eval_complexities)
        env = self._create_env(self._config, controller)
        return env, self._create_filtered_env(self._config, env)

    @staticmethod
    def _create_controller(config: ExperimentConfiguration, complexities: Sequence[int]) -> ContinuousGraphController:
        task_type = config.task_type if config.task_type is not None else ExperimentTaskType.from_task(config.task)
        factory_function = config.lm_simulation_type.get_factory_function(task_type)
        language_model = factory_function(config.seed, config.extra_args)
        rnd = Random(config.seed)
        return ContinuousGraphController(
                language_model=language_model,
                generate_init_state=lambda: config.generate_init_state(rnd, complexities, config.task),
                max_depth=config.max_depth,
                max_breadth=config.max_breadth,
                divergence_cutoff_factor=config.divergence_cutoff_factor,
                max_complexity=config.max_complexity,
                max_operations=config.max_operations
        )

    @staticmethod
    def _create_env(config: ExperimentConfiguration, controller: ContinuousGraphController) -> GraphOfThoughtsEnv:
        return GraphOfThoughtsEnv(
                config.task,
                controller,
                seed=config.seed,
                reward_version=config.reward_version,
                max_steps=config.max_steps
        )

    @staticmethod
    def _create_filtered_env(
            config: ExperimentConfiguration, graph_of_thoughts_env: GraphOfThoughtsEnv
    ) -> DictObsFilterWrapper:
        return DictObsFilterWrapper(
                graph_of_thoughts_env,
                observation_filter=config.observation_filter
        )
