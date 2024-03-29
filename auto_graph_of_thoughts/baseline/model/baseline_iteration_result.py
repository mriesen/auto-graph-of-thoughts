from dataclasses import dataclass, field, replace
from typing import Dict, Any, Optional, Self

from pure_graph_of_thoughts.api.graph.operation import OperationGraphSchema
from pure_graph_of_thoughts.api.graph.thought import ThoughtGraphSchema
from pure_graph_of_thoughts.api.schema import Schema
from .baseline_meta_info import BaselineMetaInfo


@dataclass(frozen=True)
class BaselineIterationResult(Schema):
    """
    Represents the result of a single baseline strategy iteration.
    """

    graph_of_operations: OperationGraphSchema
    """The generated graph of operations"""

    graph_of_thoughts: ThoughtGraphSchema
    """The resulting graph of thoughts"""

    is_valid: bool
    """Whether the result is valid"""

    cost: float
    """The cost of the graph of operations execution"""

    iteration: int
    """The iteration of the result"""

    meta_info: Optional[BaselineMetaInfo] = field(default=None)
    """Meta information of the iteration result"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                graph_of_operations=OperationGraphSchema.from_dict(data['graph_of_operations']),
                graph_of_thoughts=ThoughtGraphSchema.from_dict(data['graph_of_thoughts']),
                is_valid=data['is_valid'],
                cost=data['cost'],
                iteration=data['iteration'],
                meta_info=data['meta_info'] if 'meta_info' in data else None
        )

    def with_meta_info(self, meta_info: BaselineMetaInfo) -> Self:
        """
        Populates the current iteration result with the given meta information.
        :param meta_info: meta information
        :return: iteration result with meta information
        """
        return replace(self, meta_info=meta_info)
