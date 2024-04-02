from uuid import uuid4

from .id import Id


def id_generator() -> Id:
    """
    Generates an ID.
    :return: new ID
    """
    return uuid4()
