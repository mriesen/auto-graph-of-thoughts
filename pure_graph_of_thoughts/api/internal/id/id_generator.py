import itertools
from typing import Callable, Dict, Type, Any


def _create_id_generator() -> Callable[[], int]:
    """
    Creates a node ID generator.
    :return: function that returns the next node ID
    """
    id_iterator = itertools.count(0)
    return lambda: next(id_iterator)


_id_generators: Dict[Type[Any], Callable[[], int]] = {}
"""The ID generators by type."""


def id_generator(cls: Type[Any]) -> Callable[[], int]:
    """
    Get the ID generator for a given type.
    :param cls: type
    :return: ID generator
    """
    if cls not in _id_generators:
        _id_generators[cls] = _create_id_generator()
    return _id_generators[cls]
