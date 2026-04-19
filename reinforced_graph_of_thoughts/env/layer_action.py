from dataclasses import dataclass, field
from typing import Optional

from pure_graph_of_thoughts.api.operation import Operation
from .action_type import ActionType


@dataclass(frozen=True)
class LayerAction:
    """
    Represents a layer action.
    """

    type: ActionType
    """Type of action"""

    operation: Optional[Operation] = field(default=None)
    """Operation of action if applicable"""
