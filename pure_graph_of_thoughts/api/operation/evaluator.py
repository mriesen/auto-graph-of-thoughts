from dataclasses import dataclass
from typing import Callable

from ..state import State
from ..thought import Thought


@dataclass(frozen=True)
class Evaluator:
    """
    Evaluator for evaluating a thought against the ground truth.
    """

    evaluate: Callable[[State, Thought], bool]
    """Evaluator function"""
