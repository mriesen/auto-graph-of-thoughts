from enum import Enum


class OperationType(Enum):
    """
    Represents a type of operation.
    """

    GENERATE = 'generate'
    AGGREGATE = 'aggregate'
    SCORE = 'score'
