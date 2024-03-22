from dataclasses import dataclass
from typing import Self

from ..node import Node
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
    def of(cls, thought: Thought) -> Self:
        """
        Creates a node for a given thought.
        :param thought: thought of the node
        :return: new node
        """
        return cls(_thought=thought, _predecessors=[], _successors=[])

    def __hash__(self) -> int:
        return super().__hash__()
