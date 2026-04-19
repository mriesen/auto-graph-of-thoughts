class EnvWrapperException(Exception):
    """
    An exception that is raised in context of an environment wrapper.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
