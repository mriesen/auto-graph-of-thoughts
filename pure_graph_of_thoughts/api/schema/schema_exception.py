class SchemaException(Exception):
    """
    An exception that is raised when a schema is malformed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
