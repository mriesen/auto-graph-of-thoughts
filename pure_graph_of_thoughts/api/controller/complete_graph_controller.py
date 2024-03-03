from collections import deque
from itertools import islice
from typing import Dict, List, Deque, Set, Sequence, Generator, Tuple

from .controller import Controller
from .controller_exception import ControllerException
from ..graph.operation import GraphOfOperations, OperationNode
from ..graph.thought import ThoughtNode, GraphOfThoughts
from ..language_model import LanguageModel
from ..operation import Operation, PromptOperation, ExecOperation, ScoreOperation, ScorePromptOperation, \
    ScoreExecOperation
from ..state import State
from ..thought import Thought


class CompleteGraphController(Controller):
    """
    Represents a controller for executing a complete graph of operations.
    The controller traverses a given graph of operations with a breadth-first approach
    and executes all operations in order.
    """

    def __init__(self, language_model: LanguageModel) -> None:
        """
        Initializes a new complete graph controller instance.
        :param language_model: language model to use
        """
        super().__init__(language_model)

    def execute_graph(self, graph_of_operations: GraphOfOperations, init_state: State) -> GraphOfThoughts:
        """
        Executes a graph of operations.
        Traverses the graph breadth-first and executes each node's operation.
        If applicable, the scoring operation is performed afterward.

        :param graph_of_operations: graph of operations
        :param init_state: initial state
        :return: all thoughts by nodes
        """
        visited: Set[OperationNode] = set()
        queue: Deque[OperationNode] = deque([graph_of_operations.root])
        input_thought_nodes_by_operation_node: Dict[OperationNode, List[ThoughtNode]] = {}
        graph_of_thoughts: GraphOfThoughts = GraphOfThoughts.from_root(
                ThoughtNode.of(Thought(state=init_state))
        )

        while queue:
            operation_node: OperationNode = queue.popleft()
            if operation_node not in visited:
                self._logger.info('Traversing node %s', operation_node.id)

                operation_root_node = graph_of_operations.root
                thought_root_node = graph_of_thoughts.root

                input_thought_nodes = input_thought_nodes_by_operation_node[
                    operation_node
                ] if operation_node is not operation_root_node else [thought_root_node]

                input_thought_nodes_by_successor_operation_node = self._process_operation_node(
                        operation_node,
                        input_thought_nodes
                )

                for successor, successor_input_thought_nodes in input_thought_nodes_by_successor_operation_node:
                    if successor not in input_thought_nodes_by_operation_node:
                        input_thought_nodes_by_operation_node[successor] = []
                    input_thought_nodes_by_operation_node[successor].extend(successor_input_thought_nodes)

                visited.add(operation_node)

            queue.extend([successor for successor in operation_node.successors if successor not in visited])

        return graph_of_thoughts

    def _process_operation_node(
            self,
            operation_node: OperationNode,
            input_thought_nodes: Sequence[ThoughtNode]
    ) -> Sequence[Tuple[OperationNode, Sequence[ThoughtNode]]]:

        input_thoughts = [
            thought_node.thought for thought_node in input_thought_nodes
        ]
        input_states = [
            thought.state for thought in input_thoughts
        ]

        output_thoughts = self._process_operation(operation_node.operation, operation_node, input_states)
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
            self, operation: Operation, node: OperationNode, input_states: Sequence[State]
    ) -> Sequence[Thought]:
        self._logger.info('Processing operation %s', operation)

        if isinstance(operation, PromptOperation):
            input_state = operation.transform_before(input_states)
            output_state = self._language_model.prompt(operation.prompt, input_state)
            output_states = operation.transform_after(output_state)
            if operation.is_scorable and operation.score_operation is not None:
                return [
                    self._process_score_operation(operation.score_operation, node, input_state, output_state)
                    for output_state in output_states
                ]
            return [Thought(state=output_state, origin=node) for output_state in output_states]
        elif isinstance(operation, ExecOperation):
            output_states = operation.execute(input_states)
            return [Thought(state=output_state, origin=node) for output_state in output_states]
        raise ControllerException(f'Operation is not supported: {type(operation)}')

    def _process_score_operation(
            self, score_operation: ScoreOperation, node: OperationNode, previous_state: State, current_state: State
    ) -> Thought:
        self._logger.debug('Processing score operation %s', score_operation)
        if isinstance(score_operation, ScorePromptOperation):
            input_state = score_operation.transform_before(current_state)
            output_state = self._language_model.prompt(score_operation.prompt, input_state)
            score = score_operation.transform_after(output_state)
            return Thought(state=current_state, score=score, origin=node)
        elif isinstance(score_operation, ScoreExecOperation):
            score = score_operation.score(previous_state, current_state)
            return Thought(state=current_state, score=score, origin=node)
        raise ControllerException(f'Score operation is not supported: {type(score_operation)}')

    @staticmethod
    def _create_input_thoughts_buckets(
            nodes: Sequence[OperationNode], thoughts: Sequence[ThoughtNode]
    ) -> Sequence[Sequence[ThoughtNode]]:
        n_inputs = [node.operation.n_inputs for node in nodes]
        thoughts_iterator = iter(thoughts)
        return [list(islice(thoughts_iterator, n_input)) for n_input in n_inputs]
