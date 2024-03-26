from abc import ABC
from dataclasses import field, dataclass

from .id_generator import id_generator
from .id import Id


class Identifiable(ABC):
    """
    Represents a data structure that is identifiable by a unique identifier.
    """
    id: Id


@dataclass(kw_only=True, eq=False)
class AutoIdentifiable(Identifiable):
    id: Id = field(default_factory=id_generator)
    """The ID of the instance for unique identification"""
