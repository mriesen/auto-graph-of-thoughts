from abc import ABC
from dataclasses import dataclass

from ..internal.id import Identifiable, Id
from ..schema import Schema


@dataclass(frozen=True, kw_only=True)
class NodeSchema(Schema, Identifiable, ABC):
    """
    Represents a node in its schematic form.
    """

    id: Id
    """The ID of the node."""
