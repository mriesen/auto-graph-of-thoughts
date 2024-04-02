from dataclasses import dataclass
from typing import Optional, Dict, Any, Self

from ..node_schema import NodeSchema
from ...internal.id import Id, id_from_str
from ...state import State


@dataclass(frozen=True, kw_only=True)
class ThoughtNodeSchema(NodeSchema):
    """
    Represents a thought node in its schematic form.
    """

    origin_id: Optional[Id]
    """The ID of the origin operation node"""

    state: State
    """The state of the thought"""

    score: Optional[float]
    """The score of the thought"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                id=id_from_str(data['id']),
                origin_id=id_from_str(data['origin_id']) if data['origin_id'] is not None else None,
                state=data['state'],
                score=data['score']
        )
