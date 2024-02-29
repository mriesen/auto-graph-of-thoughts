
class ControllerException(Exception):
    """
    An exception raised in a controller.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)