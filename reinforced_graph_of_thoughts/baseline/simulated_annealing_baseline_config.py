from dataclasses import dataclass

from .baseline_config import BaselineConfig


@dataclass(frozen=True, kw_only=True)
class SimulatedAnnealingBaselineConfig(BaselineConfig):
    """
    The configuration of a Simulated Annealing baseline strategy execution.
    """

    neighbor_regeneration_threshold: int
    """The maximum number of neighbor re-generation attempts before performing a complete re-generation."""

    cooling_factor: float
    """The cooling factor to decrease the temperature with."""
