from dataclasses import dataclass
from typing import Dict, Any, Self

from .gpt_model import GPTModel
from ...api.schema import Schema


@dataclass(frozen=True)
class GPTUsage(Schema):
    """
    Represents the usage of a GPT model.
    """

    model: GPTModel
    """The used model"""

    n_prompt_tokens: int
    """The number of prompt tokens"""

    n_completion_tokens: int
    """The number of completion tokens"""

    total_cost: float
    """The total cost"""

    currency: str
    """The currency of the cost"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.value,
            'n_prompt_tokens': self.n_prompt_tokens,
            'n_completion_tokens': self.n_completion_tokens,
            'total_cost': self.total_cost,
            'currency': self.currency
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                model=GPTModel(data['model']),
                n_prompt_tokens=data['n_prompt_tokens'],
                n_completion_tokens=data['n_completion_tokens'],
                total_cost=data['total_cost'],
                currency=data['currency']
        )
