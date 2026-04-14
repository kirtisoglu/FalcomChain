"""
This module provides a :class:`~falcomchain.graph.Graph` class that
extends the :class:`networkx.Graph` and includes some useful methods
for working with graphs representing geographic data. The class 
:class:`~falcomchain.graph.Graph` is the only part of this module that
is intended to be used directly by users of falcomchain.

The other classes and functions in this module are used internally by
falcomchain. These include the geographic manipulation functions
available in :mod:`falcomchain.graph.geo`, the adjacency functions
in :mod:`falcomchain.graph.adjacency`, and the class
:class:`~falcomchain.graph.FrozenGraph` in the file
:mod:`falcomchain.graph.graph`. See the documentation at the top
of those files for more information.
"""

from .adjacency import *
from .geo import *
from .graph import *
from .grid import Grid
from .schema import (
    NODE_ATTRIBUTES,
    EDGE_ATTRIBUTES,
    GRAPH_ATTRIBUTES,
    SchemaValidationError,
    describe_schema,
    required_node_attributes,
    validate_graph,
)
