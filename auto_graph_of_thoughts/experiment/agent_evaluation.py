import itertools
from dataclasses import dataclass
from typing import Sequence

from .agent_evaluation_summary import AgentEvaluationSummary
from .episode import Episode


@dataclass(frozen=True)
class AgentEvaluation:
    """
    Represents an evaluation of an agent.
    """
    name: str
    """The name of the evaluation"""

    n_episodes_per_complexity: int
    """The number of episodes per complexity"""

    episodes: Sequence[Episode]
    """The episodes of the evaluation"""

    @property
    def solved_rate(self) -> float:
        """The rate of solved tasks"""
        return len([episode for episode in self.episodes if episode.is_solved]) / len(self.episodes)

    @property
    def summary(self) -> AgentEvaluationSummary:
        """The summary of the agent evaluation"""
        groups = {
            complexity: [episode for episode in group]
            for complexity, group in itertools.groupby(self.episodes, lambda episode: episode.complexity)
        }
        solved_rate_by_complexity = {
            complexity: len([episode for episode in episodes if episode.is_solved]) / len(episodes) if len(episodes) > 0 else 0
            for complexity, episodes in groups.items()
        }
        return AgentEvaluationSummary(
                name=self.name,
                n_episodes_per_complexity=self.n_episodes_per_complexity,
                solved_rate_by_complexity=solved_rate_by_complexity
        )
