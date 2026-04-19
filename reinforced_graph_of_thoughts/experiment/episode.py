from dataclasses import dataclass
from typing import Dict, Any, Self

from pure_graph_of_thoughts.api.schema import Schema


@dataclass(frozen=True)
class Episode(Schema):
    index: int
    """Episode index"""

    length: int
    """Episode length"""

    complexity: int
    """The complexity of the task"""

    total_reward: float
    """Total reward"""

    is_solved: bool
    """Whether the task is solved"""

    n_operations: int
    """The number of operations"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(**data)
