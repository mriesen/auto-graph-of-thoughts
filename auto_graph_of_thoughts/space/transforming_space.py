from abc import ABC, abstractmethod
from typing import Generic, TypeVar

S = TypeVar('S')
"""The type of the source value"""

T = TypeVar('T')
"""The type of the target value"""


class TransformingSpace(ABC, Generic[S, T]):
    """
    Represents a transforming space.
    """

    @abstractmethod
    def transform(self, value: S) -> T:
        """
        Transforms the given value into its transformed type
        :param value: source value
        :return: transformed value
        """
        pass
