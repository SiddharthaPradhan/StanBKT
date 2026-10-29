Inference Methods
=================

The fit method is chosen when the model is created and applies to every KC it fits.

.. code-block:: python

   from stanbkt.models import StandardBKT
   from stanbkt.fits import FitMethod

   model = StandardBKT(fit_method=FitMethod.VB)  # or "vb"

.. list-table::
   :header-rows: 1
   :widths: 15 20 35 30

   * - ``FitMethod``
     - Options class
     - What it produces
     - Use it for
   * - ``MCMC``
     - ``MCMCFitOptions``
     - Posterior draws from NUTS, with R-hat and ESS diagnostics.
     - Final results. The default.
   * - ``VB``
     - ``VBFitOptions``
     - Draws from a variational approximation (ADVI).
     - Fast approximate posteriors on large data.
   * - ``MLE``
     - ``MLEFitOptions``
     - A single optimum (L-BFGS). With priors this is a penalized MLE, ``jacobian=True`` gives the posterior mode.
     - Quick point estimates and starting points.
   * - ``PATHFINDER``
     - ``PFFitOptions``
     - Draws from Pathfinder's variational approximation.
     - Fast approximate posteriors.

Fit options
-----------

Each options class exposes the common CmdStanPy arguments for its method as typed fields, plus ``extra_kwargs``
for anything else. A plain dict is also accepted and forwarded to CmdStanPy.

.. code-block:: python

   from stanbkt.fits import MCMCFitOptions

   opts = MCMCFitOptions(chains=4, iter_warmup=1000, iter_sampling=1000, seed=1234, adapt_delta=0.9)
   model.fit(df, stan_fit_options=opts)

Passing an options class that does not match the fit method raises a
:class:`~stanbkt.models.FitMethodMismatchError`.

Summaries by method
-------------------

``model.summary()`` adapts to the fit method. MCMC returns CmdStan's summary (mean, MCSE, standard deviation,
percentiles, ESS and R-hat). VB and Pathfinder return mean, standard deviation and percentiles of their draws. MLE
returns a single ``value`` column and ignores ``percentiles``. See :doc:`summaries`.

Prediction support
------------------

Point estimate prediction (``predict``, ``predict_smoothed``) works for every fit method. For MLE the point
estimate is the optimum itself.

Posterior prediction (``predict_posterior_*`` and ``predict_smoothed_posterior_*``) works for MCMC, VB and MLE
with either backend. For Pathfinder fits, use ``backend="numba"``: CmdStanPy's ``generate_quantities``, which the
Stan backend runs, does not accept Pathfinder fits, so the Stan backend raises a ``ValueError`` pointing to
``backend="numba"``. The numba backend predicts from the same Pathfinder draws. ``predict_posterior_stan`` and
``predict_smoothed_posterior_stan`` return raw Stan output and are not available for Pathfinder fits. For MLE, posterior prediction has a single draw, so its intervals are degenerate.

Threading
---------

The likelihood of each KC is split into slices of students with Stan's ``reduce_sum``. To run slices in parallel
within a chain, compile with ``STAN_THREADS`` and set ``threads_per_chain`` (MCMC) or ``num_threads`` (Pathfinder).

.. code-block:: python

   model = StandardBKT(cpp_compile_kwargs={"STAN_THREADS": True})
   model.fit(df, stan_fit_options=MCMCFitOptions(chains=4, parallel_chains=4, threads_per_chain=2))

``threads_per_chain > 1`` without ``STAN_THREADS`` raises a ``ValueError``. The ``grainsize`` option and the
interaction with concurrent KC fitting are covered in :doc:`performance`.

See the :doc:`../examples/05_inference_methods` notebook for a worked example.
