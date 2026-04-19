#%% md
# # Reinforced Graph of Thoughts - PPO - Task Sum List
#%%
import argparse

from reinforced_graph_of_thoughts.agent.train_agent import train_agent
from reinforced_graph_of_thoughts.experiment import generate_init_state_sum_list
from reinforced_graph_of_thoughts.tasks.sum_list import sum_list_task

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

TASK_NAME = 'sum_list'
TRAIN_COMPLEXITIES = [c for c in range(1, 32 + 1)]
EVAL_COMPLEXITIES = [c for c in range(1, 64 + 1)]

ARTIFACTS_BASE_DIR = '../notebooks/artifacts'
RESULTS_DIR = f'{ARTIFACTS_BASE_DIR}/results/agent_evaluations/rl_tasks'

model_name = train_agent(
    seed=args.seed,
    artifacts_base_dir=ARTIFACTS_BASE_DIR,
    task_name=TASK_NAME,
    task=sum_list_task,
    train_complexities=TRAIN_COMPLEXITIES,
    eval_complexities=EVAL_COMPLEXITIES,
    generate_init_state=generate_init_state_sum_list
)