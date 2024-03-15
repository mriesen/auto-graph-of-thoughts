from dataclasses import dataclass
from typing import Self

from .thought_node import ThoughtNode
from ..graph import Graph
from ...state import State
from ...thought import Thought


@dataclass(frozen=True)
class GraphOfThoughts(Graph[ThoughtNode]):

    @classmethod
    def from_init_state(cls, init_state: State) -> Self:
        source = ThoughtNode.of(
                Thought(state=init_state)
        )
        return cls.from_source(source)
