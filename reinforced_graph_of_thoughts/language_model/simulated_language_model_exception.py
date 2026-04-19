class SimulatedLanguageModelException(Exception):
    """
    An exception raised when a language model simulation fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
