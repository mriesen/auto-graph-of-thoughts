from dataclasses import field, dataclass

from .id_generator import id_generator


@dataclass(kw_only=True, eq=False)
class Identifiable:
    """
    Represents a data structure that is identifiable by a unique identifier.
    """

    _id: int = field(init=False)
    """The ID of the instance for unique identification"""

    def __post_init__(self) -> None:
        cls = type(self)
        self._id = id_generator(cls)()

    @property
    def id(self) -> int:
        """The ID of the instance"""
        return self._id
