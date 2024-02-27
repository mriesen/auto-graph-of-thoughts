from enum import Enum


class OperationType(Enum):
    """
    Represents a type of operation.
    """

    generate = 'generate'
    aggregate = 'aggregate'
    score = 'score'
