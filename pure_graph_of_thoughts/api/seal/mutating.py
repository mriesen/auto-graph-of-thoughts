from enum import Enum
from functools import wraps
from typing import TypeVar, Callable, Any, Sequence, cast

from .sealable import Sealable
from .sealed_exception import SealedException

_T = TypeVar('_T', bound=Callable[..., Any])
"""The type of the method."""


class MutationScope(Enum):
    """
    Represents the scope of a mutation method.
    """

    ALL = 'all'
    """All passed instances are mutated"""

    SELF = 'self'
    """The current instance is mutated"""


def mutating(scope: MutationScope) -> Callable[[_T], _T]:
    """
    Marks a method of a Sealable class as mutating, meaning a sealable instance is mutated.
    Based on the scope, the method's arguments are validated to be unsealed.
    If a sealable instance in the scope is sealed, a SealedException is raised.

    :param scope: mutation scope
    :return: decorator for mutating function
    """

    def decorator(func: _T) -> _T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not args or not isinstance(args[0], Sealable):
                raise TypeError('This decorator can only be applied to methods of the Sealed class.')
            if scope == MutationScope.SELF:
                self: Sealable = args[0]
                if self.is_sealed:
                    raise SealedException(self)
            if scope == MutationScope.ALL:
                all_args: Sequence[Any] = list(args) + list(kwargs.values())
                sealable_args: Sequence[Sealable] = [arg for arg in all_args if isinstance(arg, Sealable)]
                for instance in sealable_args:
                    if instance.is_sealed:
                        raise SealedException(instance)
            return func(*args, **kwargs)

        return cast(_T, wrapper)

    return decorator
