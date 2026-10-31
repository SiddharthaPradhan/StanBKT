Troubleshooting
===============

CmdStan is not installed
------------------------

Fitting needs CmdStan. Install it once per machine with :func:`~stanbkt.utils.setup_cmdstanpy`:

.. code-block:: python

   from stanbkt.utils import setup_cmdstanpy

   setup_cmdstanpy(n_cores=4)

``setup_cmdstanpy`` raises if the installation fails.

GitHub rate limit during setup
------------------------------

CmdStan is downloaded from GitHub, which rate limits anonymous requests (common on shared networks and CI). Set a
GitHub token in the ``GITHUB_PAT`` environment variable and run the setup again, see :doc:`../installation`.

Compiler issues on Windows
--------------------------

On Windows ``setup_cmdstanpy`` installs RTools, which provides the C++ compiler and ``make``. Importing
``stanbkt.utils`` adds RTools to ``PATH`` when it is found. If compilation still fails, see the
`CmdStanPy installation guide <https://mc-stan.org/cmdstanpy/installation.html>`__.

"threads per chain specified, but STAN_THREADS ..."
---------------------------------------------------

``threads_per_chain > 1`` needs a model compiled with threading:
``StandardBKT(cpp_compile_kwargs={"STAN_THREADS": True})``. See :doc:`performance`.

"Incompatible fit options type ..."
-----------------------------------

The options class must match the model's fit method, for example ``VBFitOptions`` with ``FitMethod.VB``, otherwise
a ``FitMethodMismatchError`` is raised. Create the model with the method you want, or pass a plain dict.

"The Stan backend runs CmdStanPy's generate_quantities, which does not support Pathfinder fits"
-----------------------------------------------------------------------------------------------

CmdStanPy's ``generate_quantities`` does not accept Pathfinder fits. Use ``backend="numba"`` with the
``_draws`` and ``_summary`` posterior prediction methods. The ``_stan`` methods return raw Stan output and are not
available for Pathfinder fits. See :doc:`inference`.

"Dropping N student(s) that were not part of the fit"
-----------------------------------------------------

With ``individual_initial_knowledge=True`` and ``CORRECTNESS_ONLY``, ``pi_know`` exists only for fitted students,
so other students are dropped from predictions. Use JOINT with covariates, or population initial knowledge, to
predict new students. See :doc:`initial_knowledge`.

"... contains duplicate student IDs"
------------------------------------

The covariate or post-test frame has more than one row for a student. The first row is kept. Deduplicate the
frame to silence the warning.

"Duplicate ORDER values found within student"
---------------------------------------------

Two interactions of the same student in the same KC have the same order value, so their order is ambiguous. Use
a finer timestamp or an explicit sequence number. See :doc:`data_format`.

"Data contains no KCs that were previously fitted"
--------------------------------------------------

None of the KCs in the prediction data have a fit. Check the KC column and ``column_mapping``, and
``model.fits.get_fitted_kcs()``.

"'output_dir' cannot be passed in fit options"
----------------------------------------------

Output locations are managed by the model, because CmdStan file names only differ by the second and KCs sharing a
folder would overwrite each other's results. Use the ``temp_dir`` argument of the model. See :doc:`performance`.

"No space left on device" or memory running out during fitting
--------------------------------------------------------------

CmdStan's CSV files may be filling a temp dir that is held in RAM. Pass ``temp_dir`` with a folder on a large disk,
or set ``TMPDIR``. See :doc:`performance`.

High R-hat or low ESS
---------------------

The chains have not converged, often because a KC has little data or flat priors. Run more iterations, use
informative priors (:doc:`priors`), or check the KC with :func:`~stanbkt.plot.plot_trace`. See :doc:`summaries`.
