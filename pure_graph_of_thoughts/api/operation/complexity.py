from fractions import Fraction
from typing import Union, Optional

AbsoluteComplexity = int
"""Absolute complexity represented by a concrete number."""

RelativeComplexity = Fraction
"""Relative complexity represented by a fraction."""

Complexity = Union[AbsoluteComplexity, RelativeComplexity]
"""The complexity of a state."""


def absolute_complexity(complexity: int) -> AbsoluteComplexity:
    """
    Returns the absolute complexity.
    :param complexity: complexity
    :return: complexity
    """
    return complexity


def relative_complexity(numerator: int, denominator: Optional[int] = None) -> RelativeComplexity:
    """
    Returns the relative complexity by a given fraction.
    :param numerator: numerator
    :param denominator: denominator
    :return: relative complexity
    """
    return Fraction(numerator, denominator)


keep_complexity: RelativeComplexity = Fraction(1, 1)
"""Represents an unchanging complexity."""
