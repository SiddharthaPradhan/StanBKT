Quick Start
===========

This page shows the fastest path to install StanBKT, fit a model, and generate
knowledge-state predictions.

Requirements
------------

- Python 3.11+
- g++ compiler for Stan model compilation
- TODO add a note for windows users about using the inbuild functionality

Install
-------

From the repository root:

.. code-block:: bash

   pip install stanbkt

Or with ``uv``:

.. code-block:: bash

   uv add stanbkt

Prepare your data
-----------------

StanBKT expects interaction data in long format with these columns:

- ``student_id``: user identifier
- ``problem_id``: problem identifier
- ``order``: the order of the interaction for that student
- ``correct``: binary outcome (0/1)
- ``kc_id``: knowledge component identifier

If your column names differ, pass a ``column_mapping`` dictionary with keys:
``student_id``, ``problem_id``, ``correct``, ``order``, and ``kc_id``. To avoid hardcoding column names, use the ``ColumnNames`` constants.
Example for ASSISTments data:

.. code-block:: python

   # FIXME: update with order Id and actual assistments mappings
   from stanbkt.utils import ColumnNames
   .. code-block:: python
   :emphasize-lines: 4,5,6,7,8,9,13,15
  
   from stanbkt.utils import ColumnNames

   # Create a dict mapping expected column names to actual column names in the data
   col_mapping = {
         ColumnNames.STUDENT_ID: "user_id",
         ColumnNames.PROBLEM_ID: "item_id",
         ColumnNames.CORRECTNESS: "is_correct",
         ColumnNames.ORDER: "timestamp",
      }
   # by default, models will use MCMC for estimation.
   model = StandardBKT()
   # fit to data
   model.fit(assistments_df, column_mapping=col_mapping)


Minimal example:
We can simulate a small dataset using the inbuilt data generation utilities.


.. code-block:: python

   import pandas as pd
   from stanbkt.utils import sim_simple_bkt_data
   data_df = sim_simple_bkt_data(num_students=100, num_kcs=5, num_problems=50)
   print(data_df.head())



Fit and predict
---------------

.. code-block:: python

   from stanbkt.models.core.standard import StandardBKT

   model = StandardBKT()
   model.fit(data, method="sample")

   # Posterior summaries for p(hidden_t | correct_{1:t})
   hidden_state_summary = model.predict(data)
   print(hidden_state_summary.head())



Next steps
----------

- See API docs for model-specific details and parameters.
- Explore notebook workflows in the repository root (for example,
  ``demonstration.ipynb`` and ``model_test.ipynb``).
