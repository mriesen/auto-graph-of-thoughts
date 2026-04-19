import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.vector.utils import spaces


class QLearning:
    """
    Implementation of the Q-learning algorithm.
    """

    _env: Env[np.int64, np.int64]
    """The environment"""

    _seed: int
    """The seed for the random number generator"""

    _np_random: np.random.Generator
    """The random number generator"""

    _alpha: float
    """The learning rate"""

    _gamma: float
    """The discount factor"""

    _epsilon: float
    """The exploration rate"""

    _Q: npt.NDArray[np.float32]
    """The Q table"""

    @property
    def Q(self) -> npt.NDArray[np.float32]:
        """The Q table"""
        return self._Q

    def __init__(self, env: Env[np.int64, np.int64], alpha: float, gamma: float, epsilon: float, seed: int) -> None:
        """
        Instantiates a new Q-learning algorithm.
        :param env: environment
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: exploration rate
        :param seed: seed
        """
        self._env = env
        self._seed = seed
        self._np_random = np.random.default_rng(seed=seed)
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        if not isinstance(self._env.observation_space, spaces.Discrete):
            raise QLearningException('Observation space must be discrete')
        if not isinstance(self._env.action_space, spaces.Discrete):
            raise QLearningException('Action space must be discrete')
        observation_space: spaces.Discrete = self._env.observation_space
        action_space: spaces.Discrete = self._env.action_space
        action_space.seed(self._seed)
        self._Q = np.zeros((observation_space.n, action_space.n), dtype=np.float32)
        self._env.reset(seed=seed)

    def learn(self, total_episodes: int, verbose: bool = False) -> None:
        """
        Executes the learning process for a given number of episodes.
        :param total_episodes: total episodes to train on
        :param verbose: whether to log verbosely
        """

        for episode in range(total_episodes):
            terminated = False
            truncated = False
            total_rewards = 0.0
            state, _ = self._env.reset()

            while not terminated and not truncated:
                if self._np_random.uniform(0, 1) < self._epsilon:
                    # exploration
                    action = self._env.action_space.sample()
                else:
                    # exploitation
                    action = np.argmax(self._Q[state])
                # act
                new_state, reward, terminated, truncated, info = self._env.step(action)
                reward = float(reward)

                # update Q table
                self._Q[state, action] = self._Q[state, action] + self._alpha * (
                        reward + self._gamma * np.max(self._Q[new_state]) - self._Q[state, action]
                )

                state = new_state
                total_rewards += reward

            if verbose and episode % 100 == 0:
                print(f"Episode: {episode}, Total Reward: {total_rewards}")
        self._env.close()

    def predict(self, state: np.int64) -> np.int64:
        """
        Predicts the best action at a given state.
        :param state: current state
        :return: action
        """
        return np.argmax(self._Q[state])


class QLearningException(Exception):
    """
    An exception that is raised in context of the Q-learning algorithm.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
