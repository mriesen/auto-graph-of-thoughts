from typing import Any, Mapping, Dict, Self

import numpy as np
from gymnasium import spaces, Space

from ..obs import ObservationComponent
from .transforming_space import TransformingSpace


class MultiSpace(TransformingSpace[Mapping[ObservationComponent, Any], Mapping[str, Any]], spaces.Dict):
    """
    Represents a multi-space consisting of mappings of transforming spaces.
    The underlying Gymnasium space is the Dict space.
    """

    @classmethod
    def of(cls, all_spaces: Mapping[ObservationComponent, Any], seed: int | np.random.Generator) -> Self:
        return cls(cls._transform_str_keys(all_spaces), seed)

    @staticmethod
    def _transform_str_keys(all_spaces: Mapping[ObservationComponent, Space[Any]]) -> Dict[str, Space[Any]]:
        return {
            key.value: space for key, space in all_spaces.items()
        }

    def transform(self, value: Mapping[ObservationComponent, Any]) -> Mapping[str, Any]:
        return {
            key.value: self._transform_single(key, value) for key, value in value.items()
        }

    def _transform_single(self, key: ObservationComponent, value: Any) -> Any:
        space = self.spaces[key.value]
        if isinstance(space, TransformingSpace):
            return space.transform(value)
        return value
