from dataclasses import dataclass
from typing import Mapping, Dict, Any, Self

from pure_graph_of_thoughts.api.schema import Schema


@dataclass(frozen=True)
class AgentEvaluationSummary(Schema):
    """
    Represents the summary of an agent evaluation.
    """
    name: str
    """The name of the evaluated system"""

    n_episodes_per_complexity: int
    """The number of episodes per complexity"""

    solved_rate_per_complexity: Mapping[int, float]
    """The rate of solved tasks by complexity"""

    avg_n_operations_per_complexity: Mapping[int, float]
    """The average number of operations per complexity"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(**data)
