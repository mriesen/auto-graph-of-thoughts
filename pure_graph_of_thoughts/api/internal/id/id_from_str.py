from uuid import UUID

from .id import Id


def id_from_str(id: str) -> Id:
    """
    Converts an ID from its string representation to an ID.
    :param id: string representation of the ID
    :return: ID
    """
    return UUID(id, version=4)
