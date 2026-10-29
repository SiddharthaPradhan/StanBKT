Post-Test Outcome Regression
============================

``post_outcome_regression=True`` jointly estimates a regression of a post-test outcome on each student's final
mastery, alongside the BKT fit. Use it to ask whether mastery estimated from practice predicts an external
outcome, with the uncertainty of the BKT parameters carried into the regression.

.. note::

   Prediction methods for the post-test outcome (``predict_post_outcome*``) are not implemented yet. The regression
   coefficients are estimated and readable from ``summary()``, but there is no method that scores new students.

Model
-----

For student :math:`i`:

.. math::

   \text{post\_outcome}_i \sim \text{Normal}(b_0 + b_{mastery} \cdot \text{final\_pknow}_i + w_i b_1, \; \sigma)

``final_pknow`` is the student's mastery at time :math:`T+1`, where :math:`T` is their last interaction in the KC:
the forward filter is conditioned on all :math:`T` responses, including the last one, and then takes one more
learn/forget transition. It is the ``pKnow`` that ``predict`` would report for a next interaction. It is computed
inside the Stan model from the current BKT parameters, so it is estimated jointly rather than plugged in after
the fit.
:math:`w_i` are optional post-test covariates with slopes :math:`b_1`.

The regression is independent of the initial knowledge mode and can be combined with population, individual or
JOINT initial knowledge, in ``StandardBKT`` and ``MultiBKT``.

Data
----

.. code-block:: python

   import pandas as pd
   from stanbkt.models import StandardBKT

   post_outcome = pd.DataFrame({
       "student_id": ["stu_0", "stu_1", "stu_2"],
       "posttest": [0.8, 0.3, 0.6],
   })
   post_covariates = pd.DataFrame({
       "student_id": ["stu_0", "stu_1", "stu_2"],
       "pretest": [0.2, -1.1, 0.7],
   })

   model = StandardBKT(post_outcome_regression=True)
   model.fit(df, post_outcome=post_outcome, student_post_covariates=post_covariates)

- ``post_outcome`` has one row per student: the student ID column plus exactly one numeric outcome column. It is
  required when ``post_outcome_regression=True``.
- ``student_post_covariates`` follows the same rules as the JOINT covariate frame (:doc:`joint_strategy`) but is
  optional. Without it the outcome is regressed on final mastery alone.
- Every student in the interaction data needs a row. Missing students, NaN or infinite outcomes and a non numeric
  outcome raise a ``ValueError``. Duplicate IDs raise a warning and the first row is kept.

The outcome is modeled as normal, so it can be a score or a proportion. Its residual standard deviation has an
Exponential prior, so a very differently scaled outcome may need ``post_sigma_lambda`` adjusted.

Priors
------

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Fields
     - Prior
     - Default
   * - ``post_b0_mu``, ``post_b0_std``
     - :math:`b_0 \sim \text{Normal}`
     - 0, 5
   * - ``post_b_mastery_mu``, ``post_b_mastery_std``
     - :math:`b_{mastery} \sim \text{Normal}`
     - 0, 5
   * - ``post_b1_mu``, ``post_b1_std``
     - :math:`b_1 \sim \text{Normal}`, scalar or one value per covariate
     - 0, 5
   * - ``post_sigma_lambda``
     - :math:`\sigma \sim \text{Exponential}(\lambda)`, the rate
     - 0.5

Reading the results
-------------------

``summary()`` gains these rows when ``post_outcome_regression=True``:

- ``post_b0_param``: the intercept.
- ``post_b_mastery_param``: the change in the outcome from no mastery (``final_pknow = 0``) to full mastery (``1``).
- ``post_b1_param``: one slope per post-test covariate, labeled with the column name.
- ``post_sigma_param``: the residual standard deviation.

Each KC is a separate fit, so each KC has its own regression on its own final mastery.
