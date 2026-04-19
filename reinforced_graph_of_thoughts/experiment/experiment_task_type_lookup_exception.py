class ExperimentTaskTypeLookupException(Exception):
    """
    An exception raised when an experiment task type lookup fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
