import logging
from abc import ABC
from itertools import islice
from typing import Sequence, Tuple

from .controller_exception import ControllerException
from ..graph.operation import OperationNode
from ..graph.thought import ThoughtNode
from ..language_model import LanguageModel
from ..operation import PromptOperation, ExecOperation, ScoreOperation, ScorePromptOperation, \
    ScoreExecOperation
from ..state import State
from ..thought import Thought


class Controller(ABC):
    """
    Represents a controller for handling the execution of a graph of operations.
    """

    _language_model: LanguageModel
    _logger: logging.Logger

    def __init__(self, language_model: LanguageModel) -> None:
        """
        Initialize a new instance of a controller.
        :param language_model: language model to use
        """
        self._language_model = language_model
        self._logger = logging.getLogger(self.__class__.__name__)

    def _process_operation_node(
            self,
            operation_node: OperationNode,
            input_thought_nodes: Sequence[ThoughtNode]
    ) -> Sequence[Tuple[OperationNode, Sequence[ThoughtNode]]]:
        """
        Processes an operation node with given input thought nodes.
        :param operation_node: operation node to process
        :param input_thought_nodes: input thought nodes
        :return: mapping of thought nodes by operation node
        """

        input_thoughts = [
            thought_node.thought for thought_node in input_thought_nodes
        ]
        input_states = [
            thought.state for thought in input_thoughts
        ]

        output_thoughts = self._process_operation(operation_node, input_states)
        output_thought_nodes = [
            ThoughtNode.of(thought=thought) for thought in output_thoughts
        ]

        for input_thought_node in input_thought_nodes:
            input_thought_node.append_all(output_thought_nodes)

        successors_input_thoughts = self._create_input_thoughts_buckets(
                operation_node.successors, output_thought_nodes
        )

        return [
            (successor, successors_input_thoughts[i]) for i, successor in enumerate(operation_node.successors)
        ]

    def _process_operation(
            self, operation_node: OperationNode, input_states: Sequence[State]
    ) -> Sequence[Thought]:
        """
        Processes an operation of a given operation node with the given input states.
        :param operation_node: operation node
        :param input_states: input states
        :return: generated thoughts
        """
        operation = operation_node.operation
        self._logger.info('Processing operation %s', operation)

        if isinstance(operation, PromptOperation):
            input_state = operation.transform_before(input_states)
            output_state = self._language_model.prompt(operation.prompt, input_state)
            output_states = operation.transform_after(output_state)
            if operation.is_scorable and operation.score_operation is not None:
                return [
                    self._process_score_operation(operation.score_operation, operation_node, input_state, output_state)
                    for output_state in output_states
                ]
            return [Thought(state=output_state, origin=operation_node) for output_state in output_states]
        elif isinstance(operation, ExecOperation):
            output_states = operation.execute(input_states)
            return [Thought(state=output_state, origin=operation_node) for output_state in output_states]
        raise ControllerException(f'Operation is not supported: {type(operation)}')

    def _process_score_operation(
            self,
            score_operation: ScoreOperation,
            operation_node: OperationNode,
            previous_state: State,
            current_state: State
    ) -> Thought:
        """
        Processes a score operation for a given operation node with the previous and current state as input.
        :param score_operation: score operation to process
        :param operation_node: operation node
        :param previous_state: state of previous thought
        :param current_state: state of current thought
        :return: scored thought
        """
        self._logger.debug('Processing score operation %s', score_operation)
        if isinstance(score_operation, ScorePromptOperation):
            input_state = score_operation.transform_before(current_state)
            output_state = self._language_model.prompt(score_operation.prompt, input_state)
            score = score_operation.transform_after(output_state)
            return Thought(state=current_state, score=score, origin=operation_node)
        elif isinstance(score_operation, ScoreExecOperation):
            score = score_operation.score(previous_state, current_state)
            return Thought(state=current_state, score=score, origin=operation_node)
        raise ControllerException(f'Score operation is not supported: {type(score_operation)}')

    @staticmethod
    def _create_input_thoughts_buckets(
            operation_nodes: Sequence[OperationNode], thoughts: Sequence[ThoughtNode]
    ) -> Sequence[Sequence[ThoughtNode]]:
        """
        Creates buckets of input thoughts for each operation node.
        Given a sequence of operation nodes and thoughts, the thoughts are filled into buckets
        of the size of the operation's number of inputs.
        Each subsequence represents the input thoughts for the operation of the node at the corresponding position
        in the given sequence of operation nodes.

        :param operation_nodes: operation nodes
        :param thoughts: thoughts to fill into buckets as input
        :return: buckets of input thoughts
        """
        n_inputs = [node.operation.n_inputs for node in operation_nodes]
        thoughts_iterator = iter(thoughts)
        return [list(islice(thoughts_iterator, n_input)) for n_input in n_inputs]
