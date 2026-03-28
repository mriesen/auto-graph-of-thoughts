import re
from typing import Sequence, List, Callable, Optional

from rouge_score import rouge_scorer as _rouge_module

from pure_graph_of_thoughts.api.language_model import Prompt, Example
from pure_graph_of_thoughts.api.operation import PromptOperation, OperationType, relative_complexity, \
    absolute_complexity, ExecOperation, ScoreExecOperation
from pure_graph_of_thoughts.api.state import State
from pure_graph_of_thoughts.api.task import Task, Evaluator

# Shared ROUGE scorer instance (ROUGE-L for retention, ROUGE-1 for redundancy).
_rougeL_scorer = _rouge_module.RougeScorer(['rougeL'], use_stemmer=True)
_rouge1_scorer = _rouge_module.RougeScorer(['rouge1'], use_stemmer=True)

MergeDocsScorer = Callable[[float, State, State, Sequence[State]], float]
ComparingMergeDocsScorer = Callable[[State, Sequence[State]], float]

def _compute_retention_score(source_documents: Sequence[str], merged: str) -> float:
    """
    Measures how much information from the source documents is retained in the merged text.
    Uses ROUGE-L F1 between the merged document and the concatenation of source documents.
    1 = full retention, 0 = nothing retained.
    :param source_documents: original documents
    :param merged: merged document text
    :return: retention score in range [0, 1]
    """
    reference = ' '.join(source_documents)
    if not reference or not merged:
        return 0.0
    return _rougeL_scorer.score(reference, merged)['rougeL'].fmeasure


def _compute_non_redundancy_score(merged: str) -> float:
    """
    Measures how non-redundant the merged document is (0–1).
    Sentences are extracted and each sentence's maximum pairwise ROUGE-1 F1 with any other
    sentence is computed. The non-redundancy score is 1 minus the average of those maxima:
    1 = no sentence repeats content from another, 0 = all sentences are duplicates.
    :param merged: merged document text
    :return: non-redundancy score in range [0, 1]
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', merged) if s.strip()]
    if len(sentences) <= 1:
        return 1.0
    total_max_sim = 0.0
    for i, s1 in enumerate(sentences):
        max_sim = 0.0
        for j, s2 in enumerate(sentences):
            if i != j:
                sim = _rouge1_scorer.score(s1, s2)['rouge1'].fmeasure
                if sim > max_sim:
                    max_sim = sim
        total_max_sim += max_sim
    return 1.0 - total_max_sim / len(sentences)


def _compute_f1_score(non_redundancy: float, retention: float) -> float:
    """
    Combines non-redundancy (precision proxy) and retention (recall proxy) into an F1 score.
    :param non_redundancy: non-redundancy score (1 = no internal repetition)
    :param retention: retention score (1 = all original information retained)
    :return: F1 score in range [0, 1]
    """
    if non_redundancy + retention <= 0.0:
        return 0.0
    return 2.0 * non_redundancy * retention / (non_redundancy + retention)


def _score_state_against_others(state: State, all_states: Sequence[State]) -> float:
    """
    Scores a state containing a 'merged' document relative to its sibling states.
    Uses the concatenation of all other variants as a consensus pseudo-reference for retention,
    a variant that preserves what the majority of variants agree on is ranked higher.
    :param state: state with at least a 'merged' key
    :param all_states: all sibling states in the same keep_best operation
    :return: score in range [0, 1]
    """
    merged = state.get('merged', '')
    if not merged:
        return 0.0
    non_redundancy = _compute_non_redundancy_score(merged)
    others = [s.get('merged', '') for s in all_states if s is not state and s.get('merged')]
    retention = _compute_retention_score(others, merged) if others else non_redundancy
    return _compute_f1_score(non_redundancy, retention)


def compute_f1_score_for_merged_documents(documents: List[str], merged: str) -> float:
    return _compute_f1_score(
        _compute_non_redundancy_score(merged),
        _compute_retention_score(documents, merged)
    )


def score_op_merge_docs(
        cumulative_score: float, previous_state: State, current_state: State, output_states: Sequence[State]
) -> float:
    """
    Scores the merge_docs operation by computing the F1 of non-redundancy and retention
    of the merged document relative to the original documents.
    :param cumulative_score: cumulative score from previous operations
    :param previous_state: state before the operation (contains 'documents')
    :param current_state: state produced by the operation (contains 'merged')
    :param output_states: all output states
    :return: F1 score in range [0, 1], or -1.0 if inputs are invalid
    """
    if cumulative_score < 0.0:
        return -1.0
    merged = current_state.get('merged', '')
    documents = previous_state.get('documents', [])
    if not merged or not documents:
        return -1.0
    return compute_f1_score_for_merged_documents(documents, merged)


_MERGE_PROMPT = Prompt(
    instruction=(
        'Merge the given documents into a single comprehensive document, '
        'maximizing retained information and minimizing redundancy. '
        'Output only the merged document in JSON format with a single "merged" key.'
    ),
    examples=[
        Example(
            input={
                'documents': [
                    'Party A agrees not to disclose any confidential information received from Party B.',
                    'Party B shall not share any trade secrets of Party A with third parties.'
                ]
            },
            output={
                'merged': (
                    'Both parties agree not to disclose or share each other\'s '
                    'confidential information and trade secrets with any third parties.'
                )
            }
        )
    ]
)

_IMPROVE_PROMPT = Prompt(
    instruction=(
        'You are given an existing merged document. '
        'Improve the merged document by adding any information from the source documents that is missing '
        'and by reducing any redundancy. '
        'Output only the improved document in JSON format with a single "merged" key.'
    ),
    examples=[
        Example(
            input={
                'merged': (
                    'Party A agrees not to disclose any confidential information received from Party B.'
                    'Party B shall not share any trade secrets of Party A with third parties.'
                    'Violations of this agreement may result in legal action.'
                )
            },
            output={
                'merged': (
                    'Both parties agree not to disclose or share each other\'s '
                    'confidential information and trade secrets with any third parties. '
                    'Violations of this agreement may result in legal action.'
                )
            }
        )
    ]
)


def create_op_merge(score: MergeDocsScorer = score_op_merge_docs) -> PromptOperation:
    """
    Creates the merge operation for the merge_docs task.
    :param score: scoring function; defaults to the ROUGE-L F1 based scorer
    :return: merge PromptOperation
    """
    return PromptOperation(
        name='merge',
        n_inputs=1,
        n_outputs=1,
        type=OperationType.GENERATE,
        output_complexity=absolute_complexity(1),
        prompt=_MERGE_PROMPT,
        transform_before=lambda states: {
            'documents': states[0].get('documents', [])
        } if states else {'documents': []},
        score_operation=ScoreExecOperation(
            name='score_merge_docs',
            type=OperationType.SCORE,
            n_inputs=1,
            n_outputs=1,
            score=score
        )
    )


def create_op_improve(score: MergeDocsScorer = score_op_merge_docs) -> PromptOperation:
    """
    Creates the improve operation for the merge_docs task.
    :param score: scoring function; defaults to the ROUGE-L F1 based scorer
    :return: improve PromptOperation
    """
    return PromptOperation(
        name='improve',
        n_inputs=1,
        n_outputs=1,
        type=OperationType.GENERATE,
        output_complexity=absolute_complexity(1),
        prompt=_IMPROVE_PROMPT,
        transform_before=lambda states: {
            'merged': states[0].get('merged', '')
        } if states else {'merged': ''},
        score_operation=ScoreExecOperation(
            name='score_merge_docs',
            type=OperationType.SCORE,
            n_inputs=1,
            n_outputs=1,
            score=score
        )
    )


def create_op_keep_best_from_5(score: ComparingMergeDocsScorer = _score_state_against_others) -> ExecOperation:
    """
    Creates the keep best from 5 operation for the merge_docs task.
    :param score: scoring function; defaults to the ROUGE-L F1 based scorer
    :return: improve PromptOperation
    """
    return ExecOperation(
        name='keep_best_from_5',
        n_inputs=5,
        n_outputs=1,
        type=OperationType.AGGREGATE,
        output_complexity=relative_complexity(1),
        execute=lambda states: [max(states, key=lambda s: score(s, states), default={})]
    )


def create_merge_docs_task(score: Optional[MergeDocsScorer] = None, comparing_score: Optional[ComparingMergeDocsScorer] = None) -> Task:
    """
    Creates the merge_docs task, optionally with a custom scoring function.
    :param comparing_score: scoring function for comparison, used for keep_best_from_5, defaults to comparison against others
    :param score: scoring function for op_merge and op_improve, defaults to the ROUGE-L F1 based scorer
    :return: merge_docs Task
    """
    actual_score = score if score is not None else score_op_merge_docs
    actual_comparing_score = comparing_score if comparing_score is not None else _score_state_against_others
    op_merge_inst = create_op_merge(actual_score)
    op_improve_inst = create_op_improve(actual_score)
    op_keep_best_from_5_inst = create_op_keep_best_from_5(actual_comparing_score)
    return Task(
        operations=[
            op_merge_inst,
            op_improve_inst,
            op_branch_5,
            op_keep_best_from_5_inst
        ],
        evaluator=Evaluator(
            lambda initial_state, state: (
                    'documents' in initial_state
                    and 'merged' in state
                    and len(state['merged']) > 0
            )
        )
    )


op_branch_5 = ExecOperation(
    name='branch_5',
    output_complexity=relative_complexity(1),
    n_inputs=1,
    n_outputs=5,
    type=OperationType.GENERATE,
    execute=lambda states: [states[0] for _ in range(5)]
)

op_merge = create_op_merge()
op_improve = create_op_improve()
op_keep_best_from_5 = create_op_keep_best_from_5()

merge_docs_task = create_merge_docs_task()