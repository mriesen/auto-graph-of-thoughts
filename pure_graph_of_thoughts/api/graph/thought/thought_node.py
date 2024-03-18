from dataclasses import dataclass, field
from typing import Self

from ..node import Node, node_id_generator
from ...thought import Thought

_next_thought_node_id = node_id_generator()


@dataclass(frozen=True, eq=False)
class ThoughtNode(Node):
    """
    Represents a node in a graph of thoughts.
    """

    thought: Thought
    """The node's thought"""

    _id: int = field(default_factory=_next_thought_node_id)
    """The ID of the node for unique identification"""

    @property
    def id(self) -> int:
        """The ID of the node"""
        return self._id

    @classmethod
    def of(cls, thought: Thought) -> Self:
        """
        Creates a node for a given thought.
        :param thought: thought of the node
        :return: new node
        """
        return cls(thought=thought, _predecessors=[], _successors=[])

    def __hash__(self) -> int:
        return super().__hash__()