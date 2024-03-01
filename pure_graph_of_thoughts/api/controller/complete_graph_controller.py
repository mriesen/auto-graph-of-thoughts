from collections import deque
from itertools import islice
from typing import Dict, List, Deque, Set, Sequence, Mapping

from .controller import Controller
from .controller_exception import ControllerException
from ..graph import GraphOfOperations, Node
from ..language_model import LanguageModel
from ..operation import Operation, PromptOperation, ExecOperation, ScoreOperation, ScorePromptOperation, \
    ScoreExecOperation
from ..thought import Thought
from pure_graph_of_thoughts.api.state.state import State


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

    def execute_graph(self, graph: GraphOfOperations, init_state: State) -> Mapping[Node, Sequence[Thought]]:
        """
        Executes a graph of operations.
        Traverses the graph breadth-first and executes each node's operation.
        If applicable, the scoring operation is performed afterward.

        :param graph: graph of operations
        :param init_state: initial state
        :return: all thoughts by nodes
        """
        local_root: Node = graph.root
        visited: Set[Node] = set()
        queue: Deque[Node] = deque([local_root])
        thoughts_by_node: Dict[Node, List[Thought]] = {}
        input_thoughts_by_node: Dict[Node, List[Thought]] = {}

        while queue:
            node: Node = queue.popleft()
            if node not in visited:
                self._logger.info('Traversing node %s', node.id)
                input_states = [
                    thought.state for thought in
                    input_thoughts_by_node[node]
                ] if node is not local_root else [init_state]

                output_thoughts = self._process_operation(node.operation, node, input_states)
                thoughts_by_node[node] = list(output_thoughts)

                successors_input_thoughts = self._prepare_input_thoughts(node.successors, output_thoughts)
                for i, successor in enumerate(node.successors):
                    if successor not in input_thoughts_by_node:
                        input_thoughts_by_node[successor] = []
                    input_thoughts_by_node[successor].extend(successors_input_thoughts[i])

                visited.add(node)

            queue.extend([successor for successor in node.successors if successor not in visited])

        return thoughts_by_node

    def _process_operation(self, operation: Operation, node: Node, input_states: Sequence[State]) -> Sequence[Thought]:
        self._logger.info('Processing operation %s', operation)

        if isinstance(operation, PromptOperation):
            input_state = operation.transform_before(input_states)
            output_state = self._language_model.prompt(operation.prompt, input_state)
            output_states = operation.transform_after(output_state)
            if operation.is_scorable and operation.score_operation is not None:
                return [
                    self._process_score_operation(operation.score_operation, node, input_state, output_state) for
                    output_state in output_states
                ]
            return [Thought(state=output_state, origin=node) for output_state in output_states]
        elif isinstance(operation, ExecOperation):
            output_states = operation.execute(input_states)
            return [Thought(state=output_state, origin=node) for output_state in output_states]
        raise ControllerException(f'Operation is not supported: {type(operation)}')

    def _process_score_operation(
            self, score_operation: ScoreOperation, node: Node, previous_state: State, current_state: State
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
    def _get_thoughts_of_nodes(nodes: List[Node], thoughts_by_node: Dict[Node, List[Thought]]) -> List[Thought]:
        thoughts_of_nodes = [thought for node in nodes if node in thoughts_by_node for thought in
                             thoughts_by_node[node]]
        return thoughts_of_nodes

    @staticmethod
    def _prepare_input_thoughts(nodes: Sequence[Node], thoughts: Sequence[Thought]) -> Sequence[Sequence[Thought]]:
        n_inputs = [node.operation.n_inputs for node in nodes]
        thoughts_iterator = iter(thoughts)
        return [list(islice(thoughts_iterator, n_input)) for n_input in n_inputs]
