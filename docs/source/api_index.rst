API reference
=============

The public API is grouped by responsibility. Most workflows use
``data_conduit.datastructures`` to select sessions and assemble streams, then
add readers from ``data_conduit.datasources`` or ``data_conduit.integrations``.
Lower-level loading, time alignment, and array helpers live under
``data_conduit.core``.

.. toctree::
   :maxdepth: 2

   api/core
   api/datasources
   api/datastructures
   api/integrations
