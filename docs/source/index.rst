data-conduit
============

**Reusable data pipelines for multi-stream experimental data**

``data-conduit`` separates session discovery, source-specific loading,
cross-source configuration, and cross-session combination. Its core workflow
produces named pandas and xarray streams while retaining session provenance.

.. warning::

   The project is in alpha. Public interfaces and documentation may change
   before a stable release.

.. image:: _static/images/data-flow.svg
   :alt: Data flow through the data-conduit orchestration layer
   :width: 100%

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   overview
   workflow
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_index

Index and search
----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
