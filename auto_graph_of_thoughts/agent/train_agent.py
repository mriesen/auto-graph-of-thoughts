from random import Random

from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task

from typing import Callable, Sequence, Mapping, Any, Optional, Tuple

from stable_baselines3.common.utils import set_random_seed

from ..env import GraphObservationComponent
from ..env.create_vec_env import create_vec_env
from ..experiment import ExperimentConfiguration, LanguageModelSimulationType, Experiment, \
    evaluate_agent
from ..experiment.evaluation_summary_utils import store_evaluation_summary
from ..experiment.experiment_task_type import ExperimentTaskType
from .experiment_params import MAX_STEPS, MAX_DEPTH, MAX_BREADTH, DIVERGENCE_CUTOFF_FACTOR, MAX_OPERATIONS, \
    REWARD_VERSION, TRAIN_N_TIMESTEPS, EVAL_N_EPISODES
from .rl_model_params import POLICY_KWARGS, CLIP_RANGE, ENT_COEF, N_EPOCHS, LEARNING_RATE, N_VEC_ENVS
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import torch

SEED_PREFIX = 's'
BASE_NAME = 'ppo_r7_pi64x64_vf64x64_c1to32_t2xx18_lrlin'

MODELS_DIR = 'models/rl_tasks'
RESULTS_DIR = 'results/agent_evaluations/rl_tasks'
TB_LOG_DIR = 'tensorboard/rl_tasks'

DEVICE = 'cpu'


def _construct_name(task_name: str, seed: int) -> str:
    return f'{BASE_NAME}_{task_name}_{SEED_PREFIX}{seed}'


def train_agent(
        seed: int,
        artifacts_base_dir: str,
        task_name: str,
        task: Task,
        train_complexities: Sequence[int],
        eval_complexities: Sequence[int],
        generate_init_state: Callable[[Random, Sequence[int], Task], Tuple[int, State]],
        task_type: Optional[ExperimentTaskType] = None,
        extra_args: Optional[Mapping[str, Any]] = None
) -> str:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    config = ExperimentConfiguration(
        seed=seed,
        task=task,
        max_steps=MAX_STEPS,
        observation_filter={
            GraphObservationComponent.DEPTH,
            GraphObservationComponent.BREADTH,
            GraphObservationComponent.COMPLEXITY,
            GraphObservationComponent.PREV_ACTIONS,
            GraphObservationComponent.GRAPH_OPERATIONS,
            GraphObservationComponent.LOCAL_COMPLEXITY,
            GraphObservationComponent.PREV_SCORE
        },
        max_depth=MAX_DEPTH,
        max_breadth=MAX_BREADTH,
        divergence_cutoff_factor=DIVERGENCE_CUTOFF_FACTOR,
        train_complexities=train_complexities,
        eval_complexities=eval_complexities,
        max_complexity=max(eval_complexities),
        max_operations=MAX_OPERATIONS,
        lm_simulation_type=LanguageModelSimulationType.REALISTIC,
        reward_version=REWARD_VERSION,
        generate_init_state=generate_init_state,
        task_type=task_type,
        extra_args=extra_args if extra_args is not None else {}
    )
    experiment = Experiment(config)

    vec_env = create_vec_env(experiment.create_filtered_train_env, n_envs=N_VEC_ENVS, seed=seed)

    model = PPO(
        'MultiInputPolicy',
        vec_env,
        policy_kwargs=POLICY_KWARGS,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=experiment.config.seed,
        verbose=1,
        tensorboard_log=f'{artifacts_base_dir}/{TB_LOG_DIR}',
        device=DEVICE
    )

    model_name = _construct_name(task_name, seed)

    set_random_seed(seed)
    model.learn(total_timesteps=TRAIN_N_TIMESTEPS, tb_log_name=model_name)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=EVAL_N_EPISODES)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')
    model.save(f'{artifacts_base_dir}/{MODELS_DIR}/{model_name}')

    evaluation = evaluate_agent(
        experiment,
        model_name,
        EVAL_N_EPISODES,
        lambda obs: model.predict(obs)[0]
    )
    store_evaluation_summary(f'{artifacts_base_dir}/{RESULTS_DIR}', evaluation.summary)

    return model_name
