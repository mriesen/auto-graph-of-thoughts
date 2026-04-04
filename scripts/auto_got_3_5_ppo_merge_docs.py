#%% md
# # Automated Graph of Thoughts - PPO - Task Merge Docs
#%%
import argparse
from typing import Sequence

import pandas as pd
from pure_graph_of_thoughts.api.state import State

from auto_graph_of_thoughts.experiment.experiment_task_type import ExperimentTaskType
from auto_graph_of_thoughts.tasks.merge_docs import create_merge_docs_task

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

documents_url = 'https://raw.githubusercontent.com/spcl/graph-of-thoughts/refs/heads/main/examples/doc_merge/documents.csv'
documents = pd.read_csv(documents_url)
documents['id'] = documents['id'].astype(int)
all_documents = documents[['document1', 'document2', 'document3', 'document4']].values.tolist()


def score_has_content(
        cumulative_score: float, previous_state: State, current_state: State, output_states: Sequence[State]
) -> float:
    if cumulative_score < 0.0:
        return -1.0
    return 1.0 if current_state.get('merged') else -1.0

def score_compare_has_content(state: State, _: Sequence[State]) -> float:
    return 1.0 if state.get('merged') else -1.0

merge_docs_task = create_merge_docs_task(score=score_has_content, comparing_score=score_compare_has_content)
#%%
from auto_graph_of_thoughts.agent.train_agent import train_agent
from auto_graph_of_thoughts.experiment.generate_init_state_merge_docs import create_generate_init_state_merge_docs

TASK_NAME = 'merge_docs'

TRAIN_COMPLEXITIES = [c for c in range(1, 2 + 1)]
EVAL_COMPLEXITIES = [c for c in range(1, 4 + 1)]

ARTIFACTS_BASE_DIR = '../notebooks/artifacts'
RESULTS_DIR = f'{ARTIFACTS_BASE_DIR}/results/agent_evaluations/rl_tasks'

model_name = train_agent(
    seed=args.seed,
    artifacts_base_dir=ARTIFACTS_BASE_DIR,
    task_name=TASK_NAME,
    task=merge_docs_task,
    train_complexities=TRAIN_COMPLEXITIES,
    eval_complexities=EVAL_COMPLEXITIES,
    generate_init_state=create_generate_init_state_merge_docs(all_documents),
    task_type=ExperimentTaskType.MERGE_DOCS
)