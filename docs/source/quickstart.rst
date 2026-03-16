Quick Start
===========

This page shows the fastest path to install StanBKT, fit a model, and generate
knowledge-state predictions.

Requirements
------------

- Python 3.13+
- A working CmdStan toolchain (required by ``cmdstanpy``)

Install
-------

From the repository root:

.. code-block:: bash

   pip install -e .

Or with ``uv``:

.. code-block:: bash

   uv pip install -e .

Prepare your data
-----------------

StanBKT expects interaction data in long format with these columns:

- ``student_id``: learner identifier
- ``problem_id``: item/step identifier
- ``correct``: binary outcome (0/1)
- ``kc_id``: knowledge component identifier

Minimal example:

.. code-block:: python

   import pandas as pd

   data = pd.DataFrame(
       {
           "student_id": ["s1", "s1", "s2", "s2"],
           "problem_id": [1, 2, 1, 2],
           "correct": [0, 1, 0, 1],
           "kc_id": ["fractions", "fractions", "fractions", "fractions"],
       }
   )

Fit and predict
---------------

.. code-block:: python

   from stanbkt.models.core.standard import StandardBKT

   model = StandardBKT()
   model.fit(data, method="sample")

   # Posterior summaries for p(hidden_t | correct_{1:t})
   hidden_state_summary = model.predict(data)
   print(hidden_state_summary.head())

If your column names differ, pass a ``column_mapping`` dictionary with keys:
``student_id``, ``problem_id``, ``correct``, and ``kc_id``.

Next steps
----------

- See API docs for model-specific details and parameters.
- Explore notebook workflows in the repository root (for example,
  ``demonstration.ipynb`` and ``model_test.ipynb``).
