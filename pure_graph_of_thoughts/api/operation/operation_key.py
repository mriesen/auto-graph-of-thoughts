from dataclasses import dataclass, asdict
from typing import Dict, Any, Self

from .operation_type import OperationType
from ..schema import Schema


@dataclass(frozen=True)
class OperationKey(Schema):
    """
    Represents the key of an operation.
    An operation key consists of the basic properties of an operation.
    """

    name: str
    """The name of the operation"""

    n_inputs: int
    """The number of inputs for the operation"""

    n_outputs: int
    """The number of outputs for the operation"""

    type: OperationType
    """The type of operation"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                name=data['name'],
                n_inputs=data['n_inputs'],
                n_outputs=data['n_outputs'],
                type=OperationType[data['type']]
        )
