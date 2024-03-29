from dataclasses import dataclass
from typing import Self, Optional

from .thought_node_schema import ThoughtNodeSchema
from ..node import Node
from ...internal.id import Id
from ...thought import Thought


@dataclass(kw_only=True, eq=False)
class ThoughtNode(Node):
    """
    Represents a node in a graph of thoughts.
    """

    _thought: Thought
    """The node's thought"""

    @property
    def thought(self) -> Thought:
        return self._thought

    @classmethod
    def of(cls, thought: Thought, id: Optional[Id] = None) -> Self:
        """
        Creates a node for a given thought.
        :param thought: thought of the node
        :param id: optional id of the node
        :return: new node
        """
        if id is not None:
            return cls(id=id, _thought=thought, _predecessors=[], _successors=[])
        return cls(_thought=thought, _predecessors=[], _successors=[])

    def to_schema(self) -> ThoughtNodeSchema:
        return ThoughtNodeSchema(
                id=self.id,
                origin_id=self.thought.origin_id,
                state=self.thought.state,
                score=self.thought.score
        )

    def __hash__(self) -> int:
        return super().__hash__()
