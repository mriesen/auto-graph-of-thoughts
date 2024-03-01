from collections import UserDict
from typing import Any, Dict, Optional


class State(UserDict[str, Any]):
    """
    Represents the internal state of a thought.
    """

    def __init__(self, value: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(dict=value if value is not None else {})
