from dataclasses import dataclass, field


@dataclass(kw_only=True, eq=False)
class Sealable:
    """
    Represents a sealable instance.
    A sealable instance can be sealed, which turn the instance to an immutable data structure.
    Unsealed, the instance is mutable.
    Once sealed, the instance is immutable.
    Sealing an instance is a one-way operation.
    The enforcement of the immutability must be handled in the concrete subclasses and is not enforced by default.
    """

    _is_sealed: bool = field(default=False)
    """Whether the instance is sealed"""

    @property
    def is_sealed(self) -> bool:
        """Whether the instance is sealed"""
        return self._is_sealed

    def seal(self) -> None:
        """
        Seals the instance.
        This operation cannot be undone.
        """
        self._is_sealed = True
