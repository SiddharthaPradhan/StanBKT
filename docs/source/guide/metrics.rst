Evaluation Metrics
==================

:func:`~stanbkt.utils.accuracy`, :func:`~stanbkt.utils.rmse` and :func:`~stanbkt.utils.auc` compare observed
correctness with predicted probabilities of correctness. Each takes two flat arrays.

- ``accuracy``: the fraction of responses where ``pCorrectness >= 0.5`` matches the observed outcome.
- ``rmse``: the root mean squared error between ``pCorrectness`` and the 0/1 outcome.
- ``auc``: the area under the ROC curve, matching ``sklearn.metrics.roc_auc_score``. Tied predictions count as
  half a correct ranking, which matters in BKT since with population initial knowledge every student's first
  interaction has the same prediction. It raises a ``ValueError`` if only one outcome is present.

Evaluate with filtered predictions
----------------------------------

Use the filtered methods (``predict``, ``predict_posterior_*``) for evaluation. Smoothed predictions are
conditioned on the response they predict, so their scores are optimistic.

Hold out students rather than interactions. With population initial knowledge, a fitted model predicts any
student, so split by student ID, fit on the training students and predict the test students.

.. code-block:: python

   import numpy as np
   from stanbkt.utils import accuracy, rmse, auc

   rng = np.random.default_rng(0)
   students = df["student_id"].unique()
   test_students = rng.choice(students, size=len(students) // 5, replace=False)
   train = df[~df["student_id"].isin(test_students)]
   test = df[df["student_id"].isin(test_students)]

   model.fit(train)
   preds = model.predict(test)

Per KC scores
-------------

Predictions carry the KC column, so group by it to score each KC.

.. code-block:: python

   for kc, kc_preds in preds.groupby("kc_id", observed=True):
       y, p = kc_preds["correct"], kc_preds["pCorrectness"]
       print(kc, accuracy(y, p), rmse(y, p))

For posterior predictions, score the posterior mean, for example ``pCorrectness_mean`` from
``predict_posterior_summary``.

With individualized initial knowledge under ``CORRECTNESS_ONLY``, test students are not part of the fit and are
dropped, so use the population mode or JOINT (with covariates for the test students) for held out evaluation.

See the :doc:`../examples/11_evaluation` notebook for a worked example.
