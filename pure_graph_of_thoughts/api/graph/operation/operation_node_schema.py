from dataclasses import dataclass
from typing import Dict, Any, Self

from ..node_schema import NodeSchema
from ...operation import OperationKey


@dataclass(frozen=True, kw_only=True)
class OperationNodeSchema(NodeSchema):
    """
    Represents an operation node in its schematic form.
    """

    operation_key: OperationKey
    """The key of the operation."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        operation_key = data['operation_key']
        return cls(
                id=data['id'],
                operation_key=OperationKey.from_dict(operation_key)
        )
