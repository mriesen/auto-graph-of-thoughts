class TaskException(Exception):
    """
    An exception raised when a task instance cannot be constructed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
