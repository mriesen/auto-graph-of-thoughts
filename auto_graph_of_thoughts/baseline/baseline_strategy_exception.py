class BaselineStrategyException(Exception):
    """
    An exception raised in context of a baseline strategy.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
