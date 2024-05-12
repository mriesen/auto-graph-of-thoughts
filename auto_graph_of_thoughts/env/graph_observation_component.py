from ..obs import ObservationComponent


class GraphObservationComponent(ObservationComponent):
    """
    Represents a component of an observation.
    """

    DEPTH = 'depth'
    BREADTH = 'breadth'
    COMPLEXITY = 'complexity'
    LOCAL_COMPLEXITY = 'local_complexity'
    GRAPH_OPERATIONS = 'graph_operations'
    PREV_ACTIONS = 'prev_actions'
    PREV_SCORE = 'prev_score'
