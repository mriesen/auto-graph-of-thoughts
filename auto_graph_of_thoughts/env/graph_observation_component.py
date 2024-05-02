from ..obs import ObservationComponent


class GraphObservationComponent(ObservationComponent):
    """
    Represents a component of an observation.
    """

    depth = 'depth'
    breadth = 'breadth'
    complexity = 'complexity'
    local_complexity = 'local_complexity'
    graph_operations = 'graph_operations'
    prev_actions = 'prev_actions'
    prev_score = 'prev_score'
    divergence = 'divergence'
