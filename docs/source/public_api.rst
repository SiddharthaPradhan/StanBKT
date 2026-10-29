Public API
==========

This page documents the public API that users should interact with.
All classes and functions listed here are stable and intended for public use.

.. note::
   **For complete internal API usage, see the** :doc:`internal_api` **instead.**

Models (``stanbkt.models``)
---------------------------
Model definitions and related classes, including model variants and Bayesian prior specifications.

.. currentmodule:: stanbkt.models


Variants
^^^^^^^^

The different types of models available in `stanbkt`.

Example Usage:

.. code-block:: python
   :emphasize-lines: 5
  
   from stanbkt.models import StandardBKT

   # Create an instance of the StandardBKT model
   # By default, models use MCMC for estimation.
   model = StandardBKT()

.. autosummary::
    :signatures: long
    :caption: Model Variants
    :toctree: generated

    BKTModelBase
    StandardBKT
    MultiBKT

Constructor Options
^^^^^^^^^^^^^^^^^^^

``StandardBKT`` and ``MultiBKT`` take the same constructor arguments. ``MultiBKT`` additionally expects a
group column (``ColumnNames.GROUP``) in the data passed to ``fit`` and every ``predict*`` method.

- ``fit_method``: inference method, see :ref:`Fit Method <public_api_fit_method>`.
- ``individual_initial_knowledge`` and ``init_knowledge_strategy``: per student initial knowledge, see
  :ref:`Initial Knowledge Strategy <public_api_init_knowledge>`.
- ``post_outcome_regression``: jointly estimate a post-test outcome regression, see
  :ref:`Post-Test Outcome Regression <public_api_post_outcome>`.
- ``verbose``: logging level, see :ref:`Logging Control <public_api_logging>`.
- ``stan_compile_kwargs`` and ``cpp_compile_kwargs``: forwarded to CmdStanPy as ``stanc_options`` and
  ``cpp_options``. ``STAN_THREADS`` must be set in ``cpp_compile_kwargs`` to use ``threads_per_chain > 1``.
- ``low_memory``: evict each KC's fit to disk after fitting and reload it lazily when needed.

.. code-block:: python

   from stanbkt.models import MultiBKT
   from stanbkt.fits import FitMethod

   model = MultiBKT(
       fit_method=FitMethod.MCMC,
       cpp_compile_kwargs={"STAN_THREADS": True},  # allow within chain threading
       low_memory=True,  # keep fits on disk when fitting many KCs
   )

Model Methods
^^^^^^^^^^^^^

All model variants share these methods. Each KC in the data is fitted, predicted and summarized independently.

.. code-block:: python

   # fit every KC in the data, refitting KCs that were already fitted
   model.fit(data, overwrite_kcs=True)

   # posterior summaries of the BKT parameters, indexed by [kc, parameter, axis, label]
   model.summary()

   # point-estimate predictions (numba forward pass, or forward-backward for smoothed)
   model.predict(data, point_estimate="median")
   model.predict_smoothed(data)

   # full posterior predictions: raw Stan output, draw-level DataFrames, or per-observation summaries
   model.predict_posterior_stan(data)
   model.predict_posterior_draws(data, backend="numba")
   model.predict_posterior_summary(data, quantiles=[0.05, 0.95], n_cores=4)

   # save to a compressed archive, reload with stanbkt.load_model
   model.save("my_model")

The ``predict_smoothed_posterior_*`` methods mirror the ``predict_posterior_*`` methods with forward-backward
(smoothed) states. The ``_summary`` variants avoid materializing all draws and are recommended for large data.

Full parameter documentation for each method is on the :class:`StandardBKT` page:

- :meth:`StandardBKT.fit`
- :meth:`StandardBKT.summary`
- :meth:`StandardBKT.predict`
- :meth:`StandardBKT.predict_smoothed`
- :meth:`StandardBKT.predict_posterior_stan`
- :meth:`StandardBKT.predict_posterior_draws`
- :meth:`StandardBKT.predict_posterior_summary`
- :meth:`StandardBKT.predict_smoothed_posterior_stan`
- :meth:`StandardBKT.predict_smoothed_posterior_draws`
- :meth:`StandardBKT.predict_smoothed_posterior_summary`
- :meth:`StandardBKT.save`
- :meth:`StandardBKT.set_verbosity`

The module level :func:`predict_posterior` is the shared implementation behind every ``predict*_posterior_*``
method and takes the model as its first argument.

.. autosummary::
    :signatures: long
    :caption: Posterior Prediction
    :toctree: generated

    predict_posterior

.. _public_api_init_knowledge:

Initial Knowledge Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^

Strategy enum for modeling initial knowledge behavior.

Example Usage:

.. code-block:: python
   :emphasize-lines: 1,6,11,12

   from stanbkt.models import StandardBKT, InitKnowledgeStrategy

   # instantiate a model that jointly estimates initial knowledge from correctness and student level covariates
   model = StandardBKT(
       individual_initial_knowledge=True,
       init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
   )

   # student_covariates: one row per student, a student ID column plus one or more numeric covariate columns
   # (e.g. a pretest score). Covariate columns and their order are fixed at fit time.
   model.fit(data, student_covariates=covariates_df)
   model.predict(data, student_covariates=covariates_df)

.. autosummary::
   :signatures: long
   :caption: Initial Knowledge Strategy
   :toctree: generated

   InitKnowledgeStrategy

.. _public_api_post_outcome:

Post-Test Outcome Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mirrors the JOINT initial knowledge strategy in the opposite direction: jointly estimate a
regression of a post-test outcome on each student's final (filtered) mastery probability,
optionally plus additional post-test covariates. Independent of
``individual_initial_knowledge``/``init_knowledge_strategy`` and can be combined with either.

Example Usage:

.. code-block:: python
   :emphasize-lines: 1,4,8

   from stanbkt.models import StandardBKT

   # instantiate a model that jointly estimates final mastery's effect on a post-test outcome
   model = StandardBKT(post_outcome_regression=True)

   # post_outcome: one row per student, a student ID column plus exactly one outcome column.
   # student_post_covariates is optional (zero or more additional covariate columns).
   model.fit(data, post_outcome=post_outcome_df, student_post_covariates=post_covariates_df)

   # post_b0_param, post_b_mastery_param, post_b1_param (per covariate), post_sigma_param
   # appear as additional rows in model.summary() when post_outcome_regression=True.
   model.summary()

Bayesian Priors
^^^^^^^^^^^^^^^

Bayesian prior specification.

Example Usage:

.. code-block:: python
   :emphasize-lines: 1,5,9,13

   from stanbkt.models import StandardBKT, StandardPriors
   from stanbkt.fits import FitMethod

   # create a model with the specified estimation method
   model = StandardBKT(FitMethod.MCMC)
   
   # change the prior for pi_know (i.e. probability of knowing the skill at the start) to have 
   # a mean of 0.2 and a sd of 0.1 on the logit scale.
   priors = StandardPriors(pi_know_mu=0.2, pi_know_std=0.1)
   # Fit the model to the data
   # Note: Passing a single prior object will apply the same prior to all KCs in the data.
   #       To apply different priors to different KCs, pass a dict mapping KC ID to a prior object instead. 
   model.fit(data, priors=priors)

Use ``StandardPriors`` with ``StandardBKT`` and ``MultiPriors`` with ``MultiBKT``. ``MultiPriors`` fields accept a
scalar (shared by every group) or a list with one value per group. A list must match the number of groups in
every KC it is applied to (use a per KC dict when KCs have different groups). Passing ``None`` for a field gives a
non-informative uniform prior, and ``use_defaults=False`` sets every unspecified field to ``None``.

.. code-block:: python

   from stanbkt.models import MultiBKT, MultiPriors

   # three groups: a different prior mean for learn in each group, shared default std
   priors = MultiPriors(learn_mu=[-2.0, -1.0, 0.0])
   MultiBKT().fit(grouped_data, priors=priors)

   # uniform priors on every parameter
   uniform = MultiPriors(use_defaults=False)

.. autosummary::
    :signatures: long
    :caption: Bayesian Priors
    :toctree: generated

    StandardPriors
    MultiPriors

Model Types and Errors
^^^^^^^^^^^^^^^^^^^^^^

``ModelType`` identifies the model variant. ``FitMethodMismatchError`` is raised when something does not match the
model's ``fit_method``: fit options of another method (for example ``VBFitOptions`` for an MCMC model), a fit
object from another method, or saved artifacts whose model and fit metadata disagree on the method. It subclasses
both ``ValueError`` and ``TypeError``. It is also importable from ``stanbkt.fits``.

.. autosummary::
    :signatures: long
    :caption: Model Types and Errors
    :toctree: generated

    ModelType
    FitMethodMismatchError

Fits (``stanbkt.fits``)
-----------------------
Module containing fit methods, fit configuration options, and fit result classes.

.. currentmodule:: stanbkt.fits


.. _public_api_fit_method:

Fit Method
^^^^^^^^^^
Types of inference methods available for fitting models to data. These include MCMC, Variational Inference, Maximum Likelihood Estimation, and Pathfinder.
Usage: ``model = StandardBKT(fit_method = FitMethod.MCMC)``. Alternatively, users can specify the fitting method directly e.g. ``model = StandardBKT(fit_method='mcmc')``. 

.. autosummary::
   :signatures: long
   :caption: Fit Method
   :toctree: generated
   
   FitMethod

Fit Configuration
^^^^^^^^^^^^^^^^^


Configuration options for different inference methods. These wrap around the options provided by 
:external+cmdstanpy:class:`cmdstanpy.CmdStanModel` to provide a consistent typed interface. Specifically, these are options 
that are passed to :external+cmdstanpy:meth:`cmdstanpy.CmdStanModel.sample`,
:external+cmdstanpy:meth:`cmdstanpy.CmdStanModel.optimize`,
:external+cmdstanpy:meth:`cmdstanpy.CmdStanModel.pathfinder`, or
:external+cmdstanpy:meth:`cmdstanpy.CmdStanModel.variational` depending on the chosen fit method.


.. code-block:: python
   :emphasize-lines: 4-9,13

   from stanbkt.fits import MCMCFitOptions, FitMethod
   from stanbkt.models import StandardBKT
   
   options = MCMCFitOptions(
       iter_sampling=1000,
       iter_warmup=500,
       chains=4,
       seed=42,
   )

   model = StandardBKT(FitMethod.MCMC)
   # this will raise an error if the options provided are not compatible with the chosen fit method
   model.fit(data, stan_fit_options=options)

The likelihood of each KC is split into slices of students that run in parallel when the model is compiled
with ``STAN_THREADS``. Every fit options class has a ``grainsize`` option for the number of students per
slice. The default ``-1`` picks it per KC: one slice on a single thread, otherwise about four slices per
thread. A positive value is used as given.

.. code-block:: python

   model = StandardBKT(FitMethod.MCMC, cpp_compile_kwargs={"STAN_THREADS": True})
   model.fit(data, stan_fit_options=MCMCFitOptions(threads_per_chain=4))  # grainsize=-1
   model.fit(data, stan_fit_options=MCMCFitOptions(threads_per_chain=4, grainsize=50), overwrite_kcs=True)

.. autosummary::
   :signatures: long
   :caption: Fit Configuration
   :toctree: generated
   
   BaseFitOptions
   MCMCFitOptions
   VBFitOptions
   MLEFitOptions
   PFFitOptions

``StanFitOptions`` is the type alias for any of the four options classes, used in type hints such as
``fit(stan_fit_options: StanFitOptions | dict)``. Options passed as a ``dict`` are forwarded to CmdStanPy as is.
Passing an options class that does not match the model's fit method raises a ``FitMethodMismatchError``, and
``threads_per_chain > 1`` without ``STAN_THREADS`` in ``cpp_compile_kwargs`` raises a ``ValueError``.


Fit Results
^^^^^^^^^^^

Fit result objects for different inference methods.

These classes encapsulate the results of fitting a model to data, including parameter estimates, diagnostics, and other relevant information.
They are typically not used directly by users, however they can be used to access the underlying CmdStanPy Fit objects if needed.
``model.fits`` is always a fit container of the class matching the model's fit method, empty before the first ``fit``.

.. code-block:: python

   model.fits.get_fitted_kcs()  # set of fitted KC IDs
   fit = model.fits.get_fit("kc_0")  # CmdStanMCMC, CmdStanVB, CmdStanMLE or CmdStanPathfinder
   fit.draws_pd()

.. autosummary::
   :signatures: long
   :caption: Fit Result Classes
   :toctree: generated
   

   FitBase
   MCMCFit
   MLEFit
   VBFit
   PathfinderFit

Fit Internals
^^^^^^^^^^^^^

``FitFactory`` maps a fit method to its fit class and options class and validates fit options.
``FitMetadata`` is the metadata saved alongside fit artifacts (fit method, saved fit files and summary percentiles).
Both are used internally by the models and by save/load.

.. autosummary::
   :signatures: long
   :caption: Fit Internals
   :toctree: generated

   FitFactory
   FitMetadata

Plotting (``stanbkt.plot``)
---------------------------

.. currentmodule:: stanbkt.plot

Posterior visualization of model correctness, including:
   - Posterior predictive distributions of the probability of correctness
   - Predictions sampled from the posterior predictive distribution

.. autosummary::
   :signatures: long
   :caption: Posterior Visualization Functions
   :toctree: generated
   
   plot_posterior_correctness

Parameter plots take the CmdStanPy fit for one KC and use ArviZ. ``plot_trace`` is for MCMC fits.

.. code-block:: python

   from stanbkt.plot import plot_dist, plot_trace

   fit = model.fits.get_fit("kc_0")
   plot_dist(fit, params=["learn", "guess", "slip"], ci_prob=0.9)
   plot_trace(fit, params=["learn"])

.. autosummary::
   :signatures: long
   :caption: Parameter Plots
   :toctree: generated

   plot_dist
   plot_trace

Utilities (``stanbkt.utils``)
-----------------------------

.. currentmodule:: stanbkt.utils

.. _public_api_setup_cmdstanpy:


CmdStanPy Setup
^^^^^^^^^^^^^^^^
StanBTK uses CmdStan for model compilation, fitting and inference. This utility function installs CmdStan and sets the 
appropriate environment variables. This is required for using StanBKT for the first time, but only needs to be done once per machine.
On Windows, this always install `RTools` which installs the `c++` compiler and `make` binary. See the `CmdStanPy installation <https://mc-stan.org/cmdstanpy/installation.html>`__  documentation for 
more details.

.. autosummary::
   :signatures: long
   :caption: CmdStan Setup
   :toctree: generated

   setup_cmdstanpy

Mapping Column Names
^^^^^^^^^^^^^^^^^^^^
Utility class to map expected column names to the actual user provided column names in the data.

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
   model.fit(data, column_mapping=col_mapping)
   # predict p(correct) and p(know)
   model.predict(data, column_mapping=col_mapping)

.. autosummary::
   :signatures: long
   :caption: Column Name Mapping
   :toctree: generated

   ColumnNames

.. _public_api_logging:

Logging Control
^^^^^^^^^^^^^^^

Control verbosity to specify the amount of logging information printed during model fitting and prediction.
Three levels are available: DEBUG, INFO, and WARN, listed in decreasing order of verbosity. 

.. code-block:: python
   :emphasize-lines: 2,5
  
   from stanbkt.utils import VerbosityLevel
   # Create Model with verbosity level set to WARNING
   model = StandardBKT(verbose=VerbosityLevel.WARN)
   ...
   # this can be changed anytime
   model.set_verbosity(VerbosityLevel.DEBUG)


.. autosummary::
   :caption: Logging Control
   :toctree: generated
   
   VerbosityLevel

Compilation Cache Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
StanBKT caches compiled Stan models to speed up subsequent fits and will automatically rebuild the models if the underlying Stan code changes.


.. autosummary::
   :caption: Cache
   :toctree: generated
   
   clear_stan_cache
   get_stan_model_cache_dir
   get_cache_root
   list_cached_models
   compile_stan_model

Saving and Loading
^^^^^^^^^^^^^^^^^^
``model.save(path)`` writes the fits, cached summaries and model metadata to a single archive. ``load_model``
reconstructs the model, including the fit method and constructor options. It is importable from ``stanbkt`` or ``stanbkt.utils.model_io``.

.. code-block:: python

   from stanbkt import load_model

   model.save("my_model")
   model = load_model("my_model")

.. currentmodule:: stanbkt.utils.model_io

.. autosummary::
   :signatures: long
   :caption: Saving and Loading
   :toctree: generated

   load_model

.. currentmodule:: stanbkt.utils

Simulation
^^^^^^^^^^
Simulate long format interaction data from standard or grouped BKT parameters.

.. code-block:: python

   from stanbkt.utils import sim_simple_BKT, sim_grouped_BKT

   data = sim_simple_BKT(n_students=50, n_problems=40, n_kcs=2, learn=[0.05, 0.1], rng_seed=1234)
   # grouped parameters are a scalar, one value per group, or a [group][kc] nested list
   grouped = sim_grouped_BKT(n_students=60, n_problems=40, n_kcs=1, n_groups=2, learn=[0.03, 0.08])
   per_kc = sim_grouped_BKT(n_students=60, n_problems=40, n_kcs=2, n_groups=2, learn=[[0.03, 0.05], [0.08, 0.1]])

.. autosummary::
   :signatures: long
   :caption: Simulation
   :toctree: generated

   sim_simple_BKT
   sim_grouped_BKT

Evaluation Metrics
^^^^^^^^^^^^^^^^^^
Compare observed correctness to predicted probabilities of correctness.

.. code-block:: python

   from stanbkt.utils import accuracy, rmse, auc

   preds = model.predict(test_data)
   for kc, kc_preds in preds.groupby("kc_id", observed=True):
       print(kc, auc(kc_preds["correct"], kc_preds["pCorrectness"]), rmse(kc_preds["correct"], kc_preds["pCorrectness"]))

.. autosummary::
   :signatures: long
   :caption: Evaluation Metrics
   :toctree: generated

   accuracy
   rmse
   auc

Posterior Utilities
^^^^^^^^^^^^^^^^^^^
``posterior_summary`` turns the output of ``predict_posterior_draws`` or ``predict_posterior_stan`` (and the
smoothed variants) into per-observation statistics. ``gq_to_draws`` converts raw Stan output into draw-level
DataFrames. ``summarize_draws`` and ``summary_parameter_names`` build the parameter summaries used by
``model.summary()`` for non-MCMC fits.

.. code-block:: python

   from stanbkt.utils import posterior_summary, gq_to_draws

   stan_out = model.predict_posterior_stan(data)
   draws = gq_to_draws(stan_out, data)
   posterior_summary(draws, quantiles=[0.05, 0.95])

.. autosummary::
   :signatures: long
   :caption: Posterior Utilities
   :toctree: generated

   posterior_summary
   gq_to_draws
   summarize_draws
   summary_parameter_names

Data Utilities
^^^^^^^^^^^^^^
``validate_data`` runs the input checks used by ``fit`` and ``predict*``. ``KCData`` is the per-KC bundle
(dense correctness matrix, sequence lengths and ID maps) built from long format data.

.. autosummary::
   :signatures: long
   :caption: Data Utilities
   :toctree: generated

   validate_data
   KCData
