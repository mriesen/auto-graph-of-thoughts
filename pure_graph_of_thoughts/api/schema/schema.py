from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Self, Optional

from .schema_type_map import SchemaTypeMap


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
    def from_dict(cls, data: Dict[str, Any], type_map: Optional[SchemaTypeMap] = None) -> Self:
        """
        Instantiates a schema from a dictionary.
        :param type_map: type map for type resolution
        :param data: dictionary
        :return: schema
        """
        pass


class SchemaException(Exception):
    """
    An exception that is raised when a schema is malformed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
