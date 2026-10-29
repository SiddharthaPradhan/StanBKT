Summaries and Diagnostics
=========================

``model.summary()`` returns posterior summaries of the fitted parameters for every fitted KC, or for the KCs
passed as ``kcs``.

.. code-block:: python

   summary = model.summary()  # all KCs
   summary = model.summary(kcs="kc_0", percentiles=(5, 95))

Index
-----

The index has four levels, ``[kc, parameter, axis, label]``:

- ``kc``: the KC ID (named after the KC column).
- ``parameter``: the Stan parameter name.
- ``axis``: what the parameter is indexed by, ``group_id``, ``student_id`` or ``covariate`` (named after your
  ``column_mapping``), or empty for scalars.
- ``label``: the group name, student ID or covariate column. In ``StandardBKT`` every student is in a single
  implicit group, so group parameters have the label ``1``.

.. code-block:: python

   summary.loc["kc_0"]  # one KC
   summary.loc["kc_0"].loc["learn"]  # learn for every group of kc_0
   summary.xs("learn", level="parameter")  # learn across KCs

Pass ``label_indexes=False`` for CmdStan's raw names (``learn[1]``) with a two level ``[kc, parameter]`` index.

Parameters in the summary
-------------------------

- ``lp__``: the log density, for MCMC diagnostics.
- ``logit_pi_know_group``, ``logit_learn_group``, ``logit_forget_group``, ``logit_guess_group``,
  ``logit_slip_group``: the parameters on their unconstrained (logit or half-logit) scale, where the priors apply.
- ``pi_know``, ``learn``, ``forget``, ``guess``, ``slip``: the same parameters on the probability scale. Use these
  to interpret the model.
- Under JOINT, ``pi_b0_know_param``, ``pi_b1_know_param``, ``pi_sigma_param`` and ``logit_pi_know_z``
  (:doc:`joint_strategy`). ``pi_know`` is not reported under JOINT, since it is rebuilt from the regression.
- With ``post_outcome_regression=True``, ``post_b0_param``, ``post_b_mastery_param``, ``post_b1_param`` and
  ``post_sigma_param`` (:doc:`post_outcome`).

Columns by fit method
---------------------

- **MCMC**: CmdStan's summary, ``Mean``, ``MCSE``, ``StdDev``, ``MAD``, the two percentiles, ``50%``,
  ``ESS_bulk``, ``ESS_tail``, ``ESS_bulk/s`` and ``R_hat``.
- **VB and Pathfinder**: ``mean``, ``sd``, the two percentiles and ``50%`` of the approximate draws.
- **MLE**: a single ``value`` column. ``percentiles`` is ignored with a warning.

``percentiles`` must be two values between 1 and 99, the first smaller than the second.

Caching
-------

Summaries are cached per KC after the first call and saved with the model. A call with different ``percentiles``
recomputes them. ``clear_cache=True`` forces a refresh. With ``low_memory=True`` a cached summary never reloads a
fit from disk.

MCMC diagnostics
----------------

Check the diagnostics before interpreting an MCMC fit:

- ``R_hat`` compares chains. Values above 1.01 suggest the chains have not mixed, so run longer or revisit the
  priors.
- ``ESS_bulk`` and ``ESS_tail`` are the effective number of independent draws for the center and the tails of
  the posterior. Aim for at least 100 per chain (400 for the default 4 chains) for reliable intervals.
- ``MCSE`` is the Monte Carlo error of the mean. It should be small relative to ``StdDev``.

Weakly identified KCs (few students, flat priors, or ``guess`` and ``slip`` that explain the data as well as
learning does) show up as high ``R_hat`` and low ESS. :func:`~stanbkt.plot.plot_trace` and
:func:`~stanbkt.plot.plot_dist` show the chains and marginal posteriors of one KC:

.. code-block:: python

   from stanbkt.plot import plot_trace

   plot_trace(model.fits.get_fit("kc_0"), params=["learn", "guess", "slip"])

The underlying CmdStanPy fit is available for anything else, for example ``model.fits.get_fit("kc_0").diagnose()``.

Summary utilities
-----------------

:func:`~stanbkt.utils.summarize_draws` and :func:`~stanbkt.utils.summary_parameter_names` build the VB and
Pathfinder summaries from a draws matrix and can be used directly on any draws.
:func:`~stanbkt.utils.posterior_summary` summarizes posterior *predictions* rather than parameters, see
:doc:`prediction`.

See the :doc:`../examples/10_diagnostics_and_plotting` notebook for a worked example.
