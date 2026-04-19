import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .transforming_space import TransformingSpace


class BoolSpace(TransformingSpace[bool, npt.NDArray[np.int8]], spaces.MultiBinary):
    """
    Represents a boolean space.
    Internally, a boolean is represented by a multi-binary space of size 1.
    """

    def __init__(
            self,
            seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(1, seed=seed)

    def transform(self, value: bool) -> npt.NDArray[np.int8]:
        return np.array([int(value)], dtype=np.int8)
