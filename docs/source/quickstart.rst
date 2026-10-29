Quick Start
===========

This page shows the fastest path to install StanBKT, fit a model, and generate
knowledge-state predictions.

Requirements
------------

- Python 3.12+
- A C++ compiler and ``make`` for Stan model compilation. On Windows,
  :ref:`setup_cmdstanpy <public_api_setup_cmdstanpy>` installs RTools, which provides both.

Install
-------

.. code-block:: bash

   pip install stanbkt

Or with ``uv``:

.. code-block:: bash

   uv add stanbkt

Then install CmdStan once per machine:

.. code-block:: python

   from stanbkt.utils import setup_cmdstanpy

   setup_cmdstanpy(n_cores=4)

Prepare your data
-----------------

StanBKT expects interaction data in long format with these columns:

- ``student_id``: user identifier
- ``problem_id``: problem identifier
- ``correct``: binary outcome (0/1)
- ``timestamp``: the order of the interaction for that student
- ``kc_id``: knowledge component identifier (optional, a single KC is assumed if missing)

If your column names differ, pass a ``column_mapping`` dictionary keyed by the ``ColumnNames``
constants to ``fit`` and every ``predict*`` method.

.. code-block:: python
   :emphasize-lines: 5-11

   from stanbkt.models import StandardBKT
   from stanbkt.utils import ColumnNames

   # map expected column names to the actual column names in the data
   col_mapping = {
       ColumnNames.STUDENT_ID: "user_id",
       ColumnNames.PROBLEM_ID: "item_id",
       ColumnNames.CORRECTNESS: "is_correct",
       ColumnNames.ORDER: "order_id",
       ColumnNames.KC_ID: "skill_id",
   }
   model = StandardBKT()
   model.fit(interactions_df, column_mapping=col_mapping)

Simulate data
-------------

For a quick test, simulate a small dataset with the built-in simulation utilities.

.. code-block:: python

   from stanbkt.utils import sim_simple_BKT

   data_df = sim_simple_BKT(n_students=100, n_problems=50, n_kcs=2, rng_seed=1234)
   print(data_df.head())

Fit and predict
---------------

.. code-block:: python

   from stanbkt.models import StandardBKT

   # by default, models use MCMC for estimation
   model = StandardBKT()
   model.fit(data_df)

   # posterior summaries of the BKT parameters for each KC
   print(model.summary())

   # point-estimate predictions of p(know) and p(correct) for each interaction
   predictions = model.predict(data_df)
   print(predictions.head())

Next steps
----------

- :doc:`guide/index` explains the model, priors, inference methods, and prediction paths.
- :doc:`example_root` has worked notebooks.
- :doc:`public_api` lists every public class and function.
