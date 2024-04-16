from enum import Enum


class ActionType(Enum):
    """
    Represents a type of action.
    """

    Stop = 0
    """Action type for stopping"""

    Backtrack = 1
    """Action type for backtracking"""

    AppendOperation = 2
    """Action type for appending an operation"""
