from typing import Optional

import numpy as np
from gymnasium import spaces

from .transforming_space import TransformingSpace

OPTIONAL_BOOL_REPRESENTATION = 3
ABSENT_BOOL = 2


class OptionalBoolSpace(TransformingSpace[Optional[bool], np.int64], spaces.Discrete):
    """
    Represents an optional boolean space.
    Internally, a boolean is represented by a discrete space of size 3.
    """

    def __init__(
            self,
            seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(OPTIONAL_BOOL_REPRESENTATION, seed=seed)

    def transform(self, value: Optional[bool]) -> np.int64:
        if value is None:
            return np.int64(ABSENT_BOOL)
        return np.int64(int(value))
