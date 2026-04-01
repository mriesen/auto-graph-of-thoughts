from typing import Mapping
import plotly.express as px
import pandas as pd

_COLOR_VALID = '#6acc64'

def visualize_solved_rate(name: str, solved_rate: Mapping[int, float]) -> None:
    df = pd.DataFrame(list(solved_rate.items()), columns=['cardinality', 'solved_rate'])
    fig = px.bar(
        df,
        x='cardinality',
        y='solved_rate',
        title=f'Agent Evaluation Results for {name}',
        template='simple_white',
        labels={
            'cardinality': 'list cardinality',
            'solved_rate': 'solved tasks rate'
        },
        height=400
    )
    fig.update_xaxes(dtick=1)
    fig.update_traces(marker_color=_COLOR_VALID)
    fig.show()

def visualize_avg_n_operations(name: str, avg_n_operations: Mapping[int, float]) -> None:
    df = pd.DataFrame(list(avg_n_operations.items()), columns=['cardinality', 'avg_n_operations'])
    fig = px.bar(
        df,
        x='cardinality',
        y='avg_n_operations',
        title=f'Agent Evaluation Results for {name}',
        template='simple_white',
        labels={
            'cardinality': 'list cardinality',
            'avg_n_operations': 'average number of operations'
        },
        height=400
    )
    fig.update_xaxes(dtick=1)
    fig.update_traces(marker_color=_COLOR_VALID)
    fig.show()
