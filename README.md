# Automated Graph of Thoughts
This is the official implementation of Automated Graph of Thoughts.

## Setup Guide
To run this code, Python `3.11` or newer is required.
The latest version of the package can be installed from [PyPI](https://pypi.org/project/auto-graph-of-thoughts/):
```shell
pip install auto-graph-of-thoughts
```
Alternatively, the package can be installed from source.

### Optional Dependencies
The project comes with optional dependencies which are required for some features.

#### Notebooks
Several notebooks with examples and model training are provided with the source code.
To run the notebooks, the optional `notebooks` dependencies are required.
```shell
pip install auto-graph-of-thoughts[notebooks]
```
Be aware that the notebooks are not part of the distributed package on PyPI.

## Pure Graph of Thoughts
The package `pure_graph_of_thoughts` contains a new implementation of the Graph of Thoughts concepts.
It was originally developed in this project, before it was extracted in a dedicated project: [pure-graph-of-thoughts](https://github.com/mriesen/pure-graph-of-thoughts).