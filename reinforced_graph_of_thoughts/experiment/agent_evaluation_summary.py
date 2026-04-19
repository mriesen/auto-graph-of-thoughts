from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Dict, Any, Self, Set, Sequence

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

    train_complexities: Sequence[int]
    """The training complexities"""

    eval_complexities: Sequence[int]
    """The evaluation complexities"""

    @property
    def solved_rate_train_complexities(self) -> float:
        """The rate of solved tasks for training complexities"""
        return mean([
            solved_rate for complexity, solved_rate in self.solved_rate_per_complexity.items()
            if complexity in self.train_complexities
        ])

    @property
    def solved_rate_eval_complexities(self) -> float:
        """The rate of solved tasks for evaluation complexities"""
        return mean([
            solved_rate for complexity, solved_rate in self.solved_rate_per_complexity.items()
            if complexity in self.eval_complexities
        ])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                name=data['name'],
                n_episodes_per_complexity=data['n_episodes_per_complexity'],
                solved_rate_per_complexity=cls._create_complexity_mapping(data['solved_rate_per_complexity']),
                avg_n_operations_per_complexity=cls._create_complexity_mapping(data['avg_n_operations_per_complexity']),
                train_complexities=data['train_complexities'],
                eval_complexities=data['eval_complexities']
        )

    @staticmethod
    def _create_complexity_mapping(str_complexity_mapping: Mapping[str, float]) -> Mapping[int, float]:
        return {
            int(complexity): value for complexity, value in str_complexity_mapping.items()
        }