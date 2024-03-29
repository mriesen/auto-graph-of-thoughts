from dataclasses import dataclass
from typing import Dict, Any, Self

from ..graph_schema import GraphSchema
from .operation_node_schema import OperationNodeSchema


@dataclass(frozen=True)
class OperationGraphSchema(GraphSchema[OperationNodeSchema]):
    """
    Represents a graph of operations in its schematic form.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return super()._from_dict(data, OperationNodeSchema)
