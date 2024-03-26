import itertools

from .id import Id

_id_iterator = itertools.count(0)


def id_generator() -> Id:
    """
    Generates an ID.
    :return: new ID
    """
    return next(_id_iterator)
