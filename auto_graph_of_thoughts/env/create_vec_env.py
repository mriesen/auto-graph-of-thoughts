import os
from typing import Callable, Optional, Union, Any

from gymnasium import Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def create_vec_env(
        env_factory: Callable[..., Env[Any, Any]],
        n_envs: int = 1,
        seed: Optional[int] = None,
        monitor_dir: Optional[str] = None,
        vec_env_cls: Optional[type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
) -> VecEnv:

    def make_env(rank: int) -> Callable[[], Env[Any, Any]]:
        def _init() -> Env[Any, Any]:
            env = env_factory(rank)
            if seed is not None:
                env.action_space.seed(seed + rank)
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            return env

        return _init

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i) for i in range(n_envs)])
    vec_env.seed(seed)
    return vec_env
