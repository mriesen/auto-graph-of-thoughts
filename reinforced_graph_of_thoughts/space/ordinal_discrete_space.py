import numpy as np
import numpy.typing as npt
from gymnasium.vector.utils import spaces

from .transforming_space import TransformingSpace

SCALED_LOW: float = -1.0
SCALED_HIGH: float = 1.0

ORDINAL_DISCRETE_TYPE = np.float16


class OrdinalDiscreteSpace(TransformingSpace[int, npt.NDArray[ORDINAL_DISCRETE_TYPE]], spaces.Box):
    """
    Represents an ordinal discrete space.
    An ordinal discrete value is discrete and has an ordinal relationship to other values in the space.
    Internally, an ordinal discrete space is represented as a Gymnasium Box space.
    The values are scaled to the range [-1, 1].
    """

    _n: int
    _low: int
    _high: int

    @property
    def n(self) -> int:
        """The number of discrete values in the space"""
        return self._n

    @property
    def discrete_low(self) -> int:
        """The minimum value of the discrete space"""
        return self._low

    @property
    def discrete_high(self) -> int:
        """The maximum value of the discrete space"""
        return self._high

    def __init__(
            self,
            n: int,
            seed: int | np.random.Generator | None = None,
            start: int = 0,
    ) -> None:
        self._n = n
        self._high = n - 1 + start
        self._low = start
        super().__init__(low=SCALED_LOW, high=SCALED_HIGH, dtype=ORDINAL_DISCRETE_TYPE, seed=seed)

    def transform(self, value: int) -> npt.NDArray[ORDINAL_DISCRETE_TYPE]:
        scaled_value = (((value - self._low) * (SCALED_HIGH - SCALED_LOW)) / (self._high - self._low)) + SCALED_LOW
        return np.array([scaled_value], dtype=ORDINAL_DISCRETE_TYPE)

    def inverse_transform(self, value: float) -> int:
        unscaled_value = self._low + (value - SCALED_LOW) * (self._high - self._low) / (SCALED_HIGH - SCALED_LOW)
        return int(unscaled_value)