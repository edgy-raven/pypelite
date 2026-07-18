Pypelite API Reference
======================

Pypelite builds expressive pipelines from ordinary Python without requiring a
DAG or orchestration platform. See the `project README
<https://github.com/edgy-raven/pypelite>`_ for the guide and examples.

Pipeline
--------

.. module:: pypelite

.. autofunction:: pipeline

.. autofunction:: argument_parser

.. autofunction:: checkpoint

.. autofunction:: stage

Errors
------

.. autoclass:: PypeliteError

Archives
--------

.. autoclass:: Archive

.. autoclass:: ArchiveFormat
   :members: load, dump, hash, concat

.. module:: pypelite.configs

.. autofunction:: archive

Optional Formats
----------------

The default archive configuration uses compatible formats from installed
optional packages.

.. autofunction:: default_formatters

.. autofunction:: pandas_arrow

.. autofunction:: pandas_csv

.. autofunction:: numpy_npz

.. autofunction:: xgboost_ubj

.. autofunction:: keras_weights
