from dataclasses import dataclass
from typing import Sequence, Tuple, TypeVar, Generic, Dict, Any, Self, Type, Optional

from .node_schema import NodeSchema
from ..internal.id import Id
from ..schema import Schema, SchemaException, SchemaTypeMap

N = TypeVar('N', bound=NodeSchema)
"""The node schema type"""


@dataclass(frozen=True)
class GraphSchema(Schema, Generic[N]):
    """
    Represents a graph in its schematic form.
    """

    edges: Sequence[Tuple[Id, Id]]
    """The graph's edges"""

    nodes: Sequence[N]
    """The graph's nodes"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], type_map: Optional[SchemaTypeMap] = None) -> Self:
        edges = data['edges']
        if type_map is None or NodeSchema not in type_map:
            raise SchemaException(f'GraphSchema requires a type map with a NodeSchema mapping, got {type_map}')
        node_cls: Optional[Type[N]] = type_map[NodeSchema] if issubclass(type_map[NodeSchema], NodeSchema) else None
        if node_cls is None:
            raise SchemaException('The type map entry for NodeSchema is not a subclass of NodeSchema')
        return cls(
                edges=[
                    (edge[0], edge[1]) for edge in edges
                ],
                nodes=[
                    node_cls.from_dict(node) for node in data['nodes']
                ]
        )
