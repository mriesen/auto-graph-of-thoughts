from gymnasium.envs.registration import register

from .action_type import ActionType
from .graph_observation_component import GraphObservationComponent
from .graph_of_thoughts_env import GraphOfThoughtsEnv, GraphOfThoughtsEnvException
from .graph_step_reward import GraphStepReward, GraphStepRewardException
from .graph_step_reward_version import GraphStepRewardVersion
from .layer_action import LayerAction

register(
        id='GraphOfThoughtsEnv-v0',
        entry_point='auto_graph_of_thoughts.env:GraphOfThoughtsEnv',
        max_episode_steps=100,
)
