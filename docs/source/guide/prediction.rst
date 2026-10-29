Prediction
==========

Every prediction method returns ``pKnow`` and ``pCorrectness`` per interaction (:doc:`overview`). They differ in
two ways: point estimate versus full posterior, and filtered versus smoothed hidden states. Only KCs that have a
fit are predicted; KCs in the data without a fit are skipped with a warning.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Filtered (online)
     - Smoothed (offline)
   * - Point estimate
     - ``predict``
     - ``predict_smoothed``
   * - Posterior, raw Stan output
     - ``predict_posterior_stan``
     - ``predict_smoothed_posterior_stan``
   * - Posterior, draw level
     - ``predict_posterior_draws``
     - ``predict_smoothed_posterior_draws``
   * - Posterior, summary
     - ``predict_posterior_summary``
     - ``predict_smoothed_posterior_summary``

Use the filtered methods to evaluate predictive accuracy and to emulate what a tutor knows in real time. Use the
smoothed methods to describe mastery after the fact.

The ID and correctness columns of every output are named after ``column_mapping`` (``kc_id``, ``student_id``,
``problem_id`` and ``correct`` by default), so the output lines up with your data.

Point estimate prediction
-------------------------

``predict`` and ``predict_smoothed`` reduce each parameter's posterior to one value and run the BKT recursion
once with numba. They are fast and return one DataFrame with the KC, student and problem columns, ``pKnow``,
``pCorrectness`` and the observed correctness column.

.. code-block:: python

   preds = model.predict(df, point_estimate="median", parallel=True, fast_math=True)

- ``point_estimate``: ``"mean"`` (default), ``"median"`` or ``"mode"`` of the parameter draws.
- ``parallel``: run the numba kernel in parallel over students.
- ``fast_math``: compile the kernel with ``fastmath``. Results can differ from the strict version in the last
  digits.

Point estimates ignore parameter uncertainty, so ``pKnow`` is a single number per interaction.

Posterior prediction
--------------------

The posterior methods run the recursion once per posterior draw, so every interaction gets a distribution of
``pKnow`` and ``pCorrectness``. There are three output forms:

- ``predict_posterior_stan`` returns a dict of KC ID to the raw ``CmdStanGQ`` object from Stan's
  ``generate_quantities``. ``pKnow`` and ``pCorrectness`` are flat vectors with one entry per observation,
  student by student in interaction order.
- ``predict_posterior_draws`` returns a dict of KC ID to a long DataFrame with one row per draw and
  interaction (``chain__``, ``iter__``, ``draw__``, the ID columns, ``_order``, ``pKnow`` and ``pCorrectness``).
- ``predict_posterior_summary`` returns one DataFrame with the mean, standard deviation, median and requested
  quantiles of ``pKnow`` and ``pCorrectness`` per interaction, without materializing all draws.

.. code-block:: python

   from stanbkt.utils import posterior_summary

   # summary straight away (recommended for large data)
   summ = model.predict_posterior_summary(df, quantiles=[0.05, 0.95])

   # or keep the draws, for plotting, then summarize
   draws = model.predict_posterior_draws(df)
   summ = posterior_summary(draws, quantiles=[0.05, 0.95])

   # reuse one generate_quantities run for several outputs
   stan_out = model.predict_posterior_stan(df)
   draws = model.predict_posterior_draws(df, stan_output=stan_out)

Draw level output has ``n_draws`` rows per interaction (4000 for the default MCMC settings), so it grows quickly.
Prefer the ``_summary`` methods for large data, see :doc:`performance`.

Backends
--------

The ``_draws`` and ``_summary`` methods take ``backend``:

- ``"stan"`` (default) runs Stan's ``generate_quantities`` with the stored fit.
- ``"numba"`` replays the numba recursion for each posterior draw of the parameters. It avoids writing Stan CSV
  output, which is usually faster, and it is the only backend that works for Pathfinder fits.

Both backends agree up to floating point precision. On a 200 student test fit the numba summary was several
times faster.

Parallelism
-----------

``n_cores`` on the ``_summary`` methods runs that many KCs concurrently (``-1`` uses all cores). Each worker
holds one KC's draws in memory, so peak memory grows with ``n_cores``. Keep it at 1 when a single KC is large.

Individualized initial knowledge
--------------------------------

With ``individual_initial_knowledge=True``, pass the same ``student_covariates`` frame as at fit time under JOINT.
Students that were not part of a KC's fit are dropped with a warning under ``CORRECTNESS_ONLY`` and get the
regression mean under JOINT. See :doc:`initial_knowledge` and :doc:`joint_strategy`.
