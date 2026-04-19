from typing import Any

from gymnasium import Env
from gymnasium.wrappers import TransformObservation

from ...space import OrdinalDiscreteSpace, DiscreteSpace


class OrdinalDiscreteToDiscreteObsMappingWrapper(TransformObservation):
    """
    An observation wrapper for mapping ordinal discrete observations to discrete observations.
    """

    _low: int
    _high: int

    def __init__(self, env: Env[Any, Any], ordinal_discrete_space: OrdinalDiscreteSpace) -> None:
        """
        Instantiates a new observation mapping wrapper.
        :param env: environment
        :param ordinal_discrete_space: ordinal discrete space
        """
        discrete_space = DiscreteSpace(n=ordinal_discrete_space.n, start=ordinal_discrete_space.discrete_low)
        super().__init__(env, lambda observation: ordinal_discrete_space.inverse_transform(observation.item()))
        self._low = ordinal_discrete_space.discrete_low
        self._high = ordinal_discrete_space.discrete_high
        self.observation_space = discrete_space
        self.action_space = env.action_space
