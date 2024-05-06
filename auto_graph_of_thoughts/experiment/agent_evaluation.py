import itertools
from dataclasses import dataclass
from typing import Sequence, Set
from statistics import mean

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

    train_complexities: Set[int]
    """The training complexities"""

    eval_complexities: Set[int]
    """The evaluation complexities"""

    @property
    def solved_rate_train_complexities(self) -> float:
        """The rate of solved tasks for training complexities"""
        episodes = [episode for episode in self.episodes if episode.complexity in self.train_complexities]
        return len([episode for episode in episodes if episode.is_solved]) / len(episodes)

    @property
    def solved_rate_eval_complexities(self) -> float:
        """The rate of solved tasks for evaluation complexities"""
        episodes = [episode for episode in self.episodes if episode.complexity in self.eval_complexities]
        return len([episode for episode in episodes if episode.is_solved]) / len(episodes)

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
        avg_n_operations_by_complexity = {
            complexity: mean(episode.n_operations for episode in episodes)
            for complexity, episodes in groups.items()
        }
        return AgentEvaluationSummary(
                name=self.name,
                n_episodes_per_complexity=self.n_episodes_per_complexity,
                solved_rate_per_complexity=solved_rate_by_complexity,
                avg_n_operations_per_complexity=avg_n_operations_by_complexity,
                eval_complexities=list(self.eval_complexities),
                train_complexities=list(self.train_complexities)
        )
