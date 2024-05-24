from typing import Callable, Sequence

from ..env.graph_of_thoughts_env import ObsType, ActType
from .experiment import Experiment
from .episode import Episode
from .agent_evaluation import AgentEvaluation


def evaluate_agent(
        experiment: Experiment,
        name: str,
        n_episodes_per_complexity: int,
        agent_act: Callable[[ObsType], ActType]
) -> AgentEvaluation:
    """
    Evaluates an agent.
    :param experiment: the experiment
    :param name: the name of the evaluated system
    :param n_episodes_per_complexity: the number of episodes per complexity to evaluate
    :param agent_act: the agent call
    :return: evaluated episodes
    """
    episodes = []
    for complexity in experiment.config.eval_complexities:
        env, filtered_env = experiment.created_eval_env_tuple([complexity])
        filtered_env.reset(seed=experiment.config.seed)
        for index in range(n_episodes_per_complexity):
            obs, _ = filtered_env.reset()
            complexity = env.complexity
            terminated = False
            truncated = False
            total_reward = 0.0
            n_steps = 0

            while not terminated and not truncated:
                action = agent_act(obs)
                obs, reward, terminated, truncated, _ = filtered_env.step(action)
                total_reward += float(reward)
                n_steps += 1

            episode = Episode(
                    index=index,
                    length=n_steps,
                    complexity=complexity,
                    total_reward=total_reward,
                    is_solved=env.is_solved,
                    n_operations=env.n_operations
            )
            episodes.append(episode)
    return AgentEvaluation(
            name=name,
            n_episodes_per_complexity=n_episodes_per_complexity,
            episodes=episodes,
            train_complexities=set(experiment.config.train_complexities),
            eval_complexities=set(experiment.config.eval_complexities)
    )
