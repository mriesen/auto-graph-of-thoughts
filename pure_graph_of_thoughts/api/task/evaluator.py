from dataclasses import dataclass
from typing import Callable

from ..state import State


@dataclass(frozen=True)
class Evaluator:
    """
    Evaluator for evaluating a thought against the ground truth.
    """

    evaluate: Callable[[State, State], bool]
    """Evaluator function"""
