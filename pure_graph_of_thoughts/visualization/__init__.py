try:
    from .plot_graph import plot_graph
except ImportError:
    raise ImportError(
            'The visualisation module requires the optional visualization dependencies. '
            'Install with: pip install auto-graph-of-thoughts[visualization]'
            '\n'
            'To get a hierarchical visualization, install: '
            'pip install auto-graph-of-thoughts[visualization,dot-visualization]'
    )