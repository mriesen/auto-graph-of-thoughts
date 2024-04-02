from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple, TypeVar, Generic, Dict, Any, Self, Type

from .node_schema import NodeSchema
from ..internal.id import Id, id_from_str
from ..schema import Schema

N = TypeVar('N', bound=NodeSchema)
"""The node schema type"""


@dataclass(frozen=True)
class GraphSchema(Schema, ABC, Generic[N]):
    """
    Represents a graph in its schematic form.
    """

    edges: Sequence[Tuple[Id, Id]]
    """The graph's edges"""

    nodes: Sequence[N]
    """The graph's nodes"""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Creates a graph schema from a dictionary.
        :param data: dictionary
        :return: schema
        """
        pass

    @classmethod
    def _from_dict(cls, data: Dict[str, Any], node_cls: Type[N]) -> Self:
        """
        Creates a graph schema from a dictionary.
        :param data: dictionary
        :param type_map: type map for type resolution
        :return: schema
        """
        edges = data['edges']
        return cls(
                edges=[
                    (id_from_str(edge[0]), id_from_str(edge[1])) for edge in edges
                ],
                nodes=[
                    node_cls.from_dict(node) for node in data['nodes']
                ]
        )
