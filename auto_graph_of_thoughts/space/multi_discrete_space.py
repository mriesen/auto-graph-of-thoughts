from typing import Sequence

import numpy as np
import numpy.typing as npt
from gymnasium.vector.utils import spaces

from .transforming_space import TransformingSpace


class MultiDiscreteSpace(TransformingSpace[Sequence[int], npt.NDArray[np.int64]], spaces.MultiDiscrete):
    """
    Represents a multi discrete space.
    """

    def transform(self, value: Sequence[int]) -> npt.NDArray[np.int64]:
        return np.array(value, dtype=np.int64)
