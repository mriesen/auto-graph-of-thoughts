from typing import Any


class SealedException(Exception):
    """
    An exception that is raised when a mutation operation is performed on a sealed instance.
    """

    def __init__(self, instance: Any) -> None:
        super().__init__(f'Cannot perform a mutation operation on sealed instance: {instance}')
