from dataclasses import dataclass
from typing import Sequence, Mapping

from .gpt_model import GPTModel


@dataclass(frozen=True)
class GPTCost:
    """
    Represents the usage cost of a GPT model.
    """

    model: GPTModel
    """The GPT model"""

    prompt_token_cost: float
    """The cost per prompt token"""

    completion_token_cost: float
    """The cost per completion token"""

    currency: str = '$'
    """The currency"""


_gpt_costs: Sequence[GPTCost] = [
    GPTCost(
            model=GPTModel.GPT_35_TURBO_0125,
            prompt_token_cost=0.50 / 1_000_000,
            completion_token_cost=1.5 / 1_000_000
    ),
    GPTCost(
            model=GPTModel.GPT_35_TURBO_1106,
            prompt_token_cost=1.00 / 1_000_000,
            completion_token_cost=2.0 / 1_000_000
    ),
    GPTCost(
            model=GPTModel.GPT_4_TURBO_2024_04_09,
            prompt_token_cost=10.0 / 1_000_000,
            completion_token_cost=30.00 / 1_000_000
    ),
]

gpt_costs_by_model: Mapping[GPTModel, GPTCost] = {
    gpt_cost.model: gpt_cost for gpt_cost in _gpt_costs
}
"""The cost of the GPT models by model."""
