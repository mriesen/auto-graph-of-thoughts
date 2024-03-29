from dataclasses import dataclass
from typing import Dict, Any, Self

from .thought_node_schema import ThoughtNodeSchema
from ..graph_schema import GraphSchema


@dataclass(frozen=True)
class ThoughtGraphSchema(GraphSchema[ThoughtNodeSchema]):
    """
    Represents a graph of thoughts in its schematic form.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return super()._from_dict(data, ThoughtNodeSchema)
