import numpy as np
from gymnasium import spaces

from .transforming_space import TransformingSpace


class DiscreteSpace(TransformingSpace[int, np.int64], spaces.Discrete):
    """
    Represents a discrete space.
    """

    def transform(self, value: int) -> np.int64:
        return np.int64(value)
