JOINT Initial Knowledge
=======================

``InitKnowledgeStrategy.JOINT`` estimates each student's initial knowledge from a regression on student level
covariates, jointly with the BKT parameters. Use it when you have information about students before they start,
such as a pretest score, prior grades or demographics.

Model
-----

For student :math:`i` with covariate row :math:`x_i`:

.. math::

   \text{logit}(\pi_{know,i}) &= b_0 + x_i b_1 + \sigma z_i \\
   z_i &\sim \text{Normal}(0, 1)

The residual is non-centered: :math:`z_i` is a standard normal latent per fitted student and :math:`\sigma`
scales it. :math:`b_1` has one coefficient per covariate. ``learn``, ``forget``, ``guess`` and ``slip`` are
unchanged, and in ``MultiBKT`` they stay per group while the regression is shared across groups.

The covariate frame
-------------------

Pass covariates as a separate DataFrame with one row per student: the student ID column (named as in
``column_mapping``) plus one or more numeric covariate columns. Every other column is treated as a covariate, in
the frame's own column order.

.. code-block:: python

   import pandas as pd
   from stanbkt.models import StandardBKT, InitKnowledgeStrategy

   covariates = pd.DataFrame({
       "student_id": ["stu_0", "stu_1", "stu_2"],
       "pretest": [0.2, -1.1, 0.7],
       "prior_gpa": [0.5, 0.0, -0.3],
   })

   model = StandardBKT(
       individual_initial_knowledge=True,
       init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
   )
   model.fit(df, student_covariates=covariates)

The frame is validated on every call:

- A missing student ID column, no covariate columns, non numeric covariate columns, NaN or infinite values, or a
  student in the interaction data with no row in the frame all raise a ``ValueError``.
- Duplicate student IDs raise a warning and the first row of each is kept.
- Extra students in the frame that are not in the interaction data are ignored.
- The covariate columns and their order are saved with each KC's fit. At prediction time they must match,
  otherwise a ``ValueError`` is raised.

Covariates are used as given. Center and scale them before fitting: the default prior on :math:`b_1`
(normal with standard deviation 5) assumes coefficients on roughly unit scale, and centered covariates make
:math:`b_0` the logit initial knowledge of an average student.

Priors
------

The regression has its own prior fields on :class:`~stanbkt.models.StandardPriors` and
:class:`~stanbkt.models.MultiPriors`. ``pi_know_mu`` and ``pi_know_std`` are not used under JOINT.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Fields
     - Prior
     - Default
   * - ``pi_b0_know_mu``, ``pi_b0_know_std``
     - :math:`b_0 \sim \text{Normal}`
     - 0, 5
   * - ``pi_b1_know_mu``, ``pi_b1_know_std``
     - :math:`b_1 \sim \text{Normal}`, scalar or one value per covariate
     - 0, 5
   * - ``pi_sigma_lambda``
     - :math:`\sigma \sim \text{Exponential}(\lambda)`, the rate
     - 5 (mean 0.2)

.. code-block:: python

   from stanbkt.models import StandardPriors

   # a strong positive prior on the pretest slope, a weak one on prior_gpa
   priors = StandardPriors(pi_b1_know_mu=[1.0, 0.0], pi_b1_know_std=[0.5, 2.0])
   model.fit(df, student_covariates=covariates, priors=priors, overwrite_kcs=True)

Reading the results
-------------------

``summary()`` has these rows under JOINT, and only under JOINT. Default and ``CORRECTNESS_ONLY`` fits have
exactly the same parameters as before.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - ``parameter``
     - ``axis`` and ``label``
     - Meaning
   * - ``pi_b0_know_param``
     -
     - Intercept :math:`b_0` (logit scale).
   * - ``pi_b1_know_param``
     - ``covariate``, the column name
     - One slope per covariate (logit scale).
   * - ``pi_sigma_param``
     -
     - Residual standard deviation :math:`\sigma`.
   * - ``logit_pi_know_z``
     - ``student_id``, the student ID
     - Each fitted student's standardized residual :math:`z_i`.

.. code-block:: python

   s = model.summary().loc["kc_0"]
   s.loc[["pi_b0_know_param", "pi_b1_know_param", "pi_sigma_param"]]

A slope's sign and interval tell you whether the covariate predicts higher initial knowledge. A small
:math:`\sigma` means the covariates explain most of the between student variation.

Identifiability
^^^^^^^^^^^^^^^

Each student contributes one binary piece of information about their initial knowledge: whether they started in
the known state, which the model only sees through noisy responses afterwards. A logistic regression with a per
student residual is weakly identified from that: a larger :math:`\sigma` with proportionally larger slopes predicts
almost the same probabilities. In practice:

- The prior on :math:`\sigma` decides where the fit lands. The default ``pi_sigma_lambda=5`` (mean 0.2) assumes the
  covariates explain most of the differences between students, which keeps the slopes from inflating. A loose prior
  such as ``pi_sigma_lambda=0.5`` (mean 2) allows a large :math:`\sigma` and inflated slopes, often with ``R_hat``
  above 1.01. If you expect large differences between students that the covariates do not capture, loosen the
  prior, but read the size of the slopes with care.
- With a few hundred students the slopes tend to be overestimated, and they settle as the number of students grows.
- Compare slopes with each other and with their intervals, and refit with more than one :math:`\sigma` prior when
  the size of the slopes matters for your conclusions.

The :doc:`../examples/07_joint_initial_knowledge` notebook shows this on simulated data.

Prediction
----------

Pass ``student_covariates`` to every ``predict*`` method as well.

- **Seen students** (part of the KC's fit, matched by ID) use :math:`b_0 + x_i b_1 + \sigma z_i` with their
  fitted :math:`z_i` and the covariates in the prediction frame.
- **Unseen students** use the regression mean :math:`b_0 + x_i b_1`, with no residual.

Point estimates (``predict(point_estimate=...)``) are computed from the per draw ``pi_know`` on the probability
scale. See :doc:`prediction` for the prediction methods.
