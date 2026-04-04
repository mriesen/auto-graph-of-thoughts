#%% md
# # Automated Graph of Thoughts - PPO - Task Count Keywords
#%%
import argparse

import pandas as pd
from pure_graph_of_thoughts.api.language_model import Example

from auto_graph_of_thoughts.experiment.experiment_task_type import ExperimentTaskType
from auto_graph_of_thoughts.tasks.count_keywords import create_count_keywords_task, create_op_count

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

countries_url = 'https://raw.githubusercontent.com/spcl/graph-of-thoughts/refs/heads/main/examples/keyword_counting/countries.csv'
countries = pd.read_csv(countries_url)
text_keyword_map = (
    countries.dropna(subset=['Countries'])
    .assign(Countries=lambda df: df['Countries'].apply(
        lambda x: x[1:-1].split(', ') if isinstance(x, str) and len(x) > 1 else []))
    .set_index('Text')['Countries']
    .to_dict()
)
keywords = list(set([
    item for value in text_keyword_map.values() for item in value
]))
all_texts = list(text_keyword_map.keys())

op_count_keywords = create_op_count(
    instruction='Count the occurrence of countries in the given text.',
    examples=[
        Example(
            input={
                'text': 'France and Italy are known for their rich cultural heritage and exquisite cuisine, while Japan offers a blend of ancient tradition and cutting-edge technology. Meanwhile, Italy\u2019s scenic countryside, Brazil\u2019s vibrant festivals and Australia\u2019s stunning landscapes attract travelers from around the world.'
            },
            output={
                'count': {
                    'France': 1,
                    'Italy': 2,
                    'Japan': 1,
                    'Brazil': 1,
                    'Australia': 1,
                    'Switzerland': 1
                }
            }
        )
    ],
    keywords=keywords
)
count_keywords_task = create_count_keywords_task(keywords, op_count_keywords)
#%%
from auto_graph_of_thoughts.agent.train_agent import train_agent
from auto_graph_of_thoughts.experiment import create_generate_init_state_count_keywords

TASK_NAME = 'count_keywords'

TRAIN_COMPLEXITIES = [c for c in range(10, 50 + 1, 10)]
EVAL_COMPLEXITIES = [c for c in range(10, 100 + 1, 10)]

ARTIFACTS_BASE_DIR = '../notebooks/artifacts'
RESULTS_DIR = f'{ARTIFACTS_BASE_DIR}/results/agent_evaluations/rl_tasks'

model_name = train_agent(
    seed=args.seed,
    artifacts_base_dir=ARTIFACTS_BASE_DIR,
    task_name=TASK_NAME,
    task=count_keywords_task,
    train_complexities=TRAIN_COMPLEXITIES,
    eval_complexities=EVAL_COMPLEXITIES,
    generate_init_state=create_generate_init_state_count_keywords(all_texts),
    task_type=ExperimentTaskType.COUNT_KEYWORDS,
    extra_args={
        'keywords': keywords,
        'op_count': op_count_keywords
    }
)