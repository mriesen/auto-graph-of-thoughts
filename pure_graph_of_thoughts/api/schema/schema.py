from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Self


@dataclass(frozen=True)
class Schema(ABC):
    """
    Represents an object in its schematic form.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the schema into a dictionary.
        :return: dictionary
        """
        return asdict(self)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Instantiates a schema from a dictionary.
        :param data: dictionary
        :return: schema
        """
        pass
