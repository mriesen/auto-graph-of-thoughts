from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, Dict, Any, Optional, Self

from pure_graph_of_thoughts.api.schema import Schema
from .baseline_iteration_result import BaselineIterationResult


@dataclass(frozen=True, kw_only=True)
class BaselineResultSummary(Schema):
    """
    Represents the result of a complete baseline strategy execution.
    """

    results: Sequence[BaselineIterationResult]
    """All results of the baseline strategy execution"""

    final_result_index: Optional[int]
    """The index of the final result of the baseline strategy execution"""

    max_iterations: int
    """The maximum number of iterations"""

    stop_on_first_valid: bool = field(default=False)
    """Whether the generation was stopped on the first valid iteration result"""

    created_at: datetime = field(default_factory=datetime.now)
    """The timestamp of the baseline result summary creation"""

    @property
    def final_result(self) -> Optional[BaselineIterationResult]:
        return self.results[self.final_result_index] if self.final_result_index is not None else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                results=[
                    BaselineIterationResult.from_dict(result) for result in data['results']
                ],
                final_result_index=data['final_result_index'],
                max_iterations=data['max_iterations'],
                stop_on_first_valid=data['stop_on_first_valid'],
                created_at=datetime.fromisoformat(data['created_at'])
        )
