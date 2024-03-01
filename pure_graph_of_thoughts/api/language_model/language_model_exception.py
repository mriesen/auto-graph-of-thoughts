
class LanguageModelException(Exception):
    """
    An exception raised while accessing a language model.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)