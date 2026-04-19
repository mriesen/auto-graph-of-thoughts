from pure_graph_of_thoughts.api.schema import JsonSchemaEncoder
import json
import os
from .agent_evaluation_summary import AgentEvaluationSummary


def store_evaluation_summary(results_directory: str, evaluation_summary: AgentEvaluationSummary) -> None:
    """
    Stores an evaluation summary to file.
    :param results_directory: results directory
    :param evaluation_summary: evaluation summary to store
    """
    file_name = f'{results_directory}/{evaluation_summary.name}.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, cls=JsonSchemaEncoder, ensure_ascii=False, indent=2)

def load_evaluation_summary(results_directory: str, name: str) -> AgentEvaluationSummary:
    """
    Loads an evaluation summary.
    :param results_directory: results directory
    :param name: name
    :return: loaded evaluation summary
    """
    file_name = f'{results_directory}/{name}.json'
    with open(file_name, 'r', encoding='utf-8') as f:
        return AgentEvaluationSummary.from_dict(json.load(f))
