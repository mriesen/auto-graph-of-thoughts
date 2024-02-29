import logging
from abc import ABC

from ..language_model import LanguageModel


class Controller(ABC):
    """
    Represents a controller for handling the execution of a graph of operations.
    """

    _language_model: LanguageModel
    _logger: logging.Logger

    def __init__(self, language_model: LanguageModel) -> None:
        """
        Initialize a new instance of a controller.
        :param language_model: language model to use
        """
        self._language_model = language_model
        self._logger = logging.getLogger(self.__class__.__name__)
