from enum import Enum


class ActionType(Enum):
    """
    Represents a type of action.
    """

    STOP = 0
    """Action type for stopping"""

    BACKTRACK = 1
    """Action type for backtracking"""

    APPEND_OPERATION = 2
    """Action type for appending an operation"""
