FalcomChain API
===============

falcomchain.graph
-----------------

.. autoclass:: falcomchain.graph.Graph
   :members:
   :undoc-members:

.. autoclass:: falcomchain.graph.FrozenGraph
   :members:

.. autoclass:: falcomchain.graph.Grid
   :members:

.. automodule:: falcomchain.graph.schema
   :members:

falcomchain.partition
---------------------

.. autoclass:: falcomchain.partition.Partition
   :members:
   :undoc-members:

.. autoclass:: falcomchain.partition.assignment.Assignment
   :members:

.. autoclass:: falcomchain.partition.flows.Flow
   :members:

falcomchain.tree
----------------

.. autoclass:: falcomchain.tree.tree.SpanningTree
   :members:

.. autoclass:: falcomchain.tree.tree.CutParams
   :members:

.. autoclass:: falcomchain.tree.tree.Cut
   :members:

.. autoclass:: falcomchain.tree.tree.Flip
   :members:

.. autofunction:: falcomchain.tree.tree.uniform_spanning_tree

.. autofunction:: falcomchain.tree.tree.random_spanning_tree

.. autofunction:: falcomchain.tree.tree.bipartition_tree

.. autofunction:: falcomchain.tree.tree.capacitated_recursive_tree

.. autoclass:: falcomchain.tree.snapshot.Recorder
   :members:

falcomchain.markovchain
-----------------------

.. autoclass:: falcomchain.markovchain.MarkovChain
   :members:

.. autoclass:: falcomchain.markovchain.state.ChainState
   :members:

.. autoclass:: falcomchain.markovchain.facility.FacilityAssignment
   :members:

.. autoclass:: falcomchain.markovchain.facility.SuperFacilityAssignment
   :members:

.. autofunction:: falcomchain.markovchain.proposals.hierarchical_recom

.. autofunction:: falcomchain.markovchain.accept.always_accept

.. autofunction:: falcomchain.markovchain.accept.metropolis_hastings

.. autofunction:: falcomchain.markovchain.energy.compute_energy

.. autofunction:: falcomchain.markovchain.energy.compute_energy_delta

falcomchain.constraints
-----------------------

.. autoclass:: falcomchain.constraints.Validator
   :members:

falcomchain.random
------------------

.. automodule:: falcomchain.random
   :members:
