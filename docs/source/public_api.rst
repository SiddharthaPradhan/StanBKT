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
   # by default models will use MCMC for estimation. See :
   model = StandardBKT()

.. autosummary::
    :signatures: long
    :caption: Model Variants
    :toctree: generated

    BKTModelBase
    StandardBKT
    MultiBKT

Bayesian Priors
^^^^^^^^^^^^^^^

Bayesian prior specification.

Example Usage:

.. code-block:: python
   :emphasize-lines: 5,7,11,15

   from stanbkt.models import StandardBKT, StandardPriors, InitKnowledgeStrategy
   from stanbkt.fits import FitMethod

   # select FIXME prior estimation method
   est_type = InitKnowledgeStrategy.FIXME
   # create a model with the specified estimation method
   model = StandardBKT(FitMethod.MCMC, prior_estimation_type=est_type)
   
   # change the prior for pi_know (i.e. probability of knowing the skill at the start) to have 
   # a mean of 0.2 and a sd of 0.1 on the logit scale.
   priors = BayesianPriors(pi_know_mu=0.2, pi_know_sigma=0.1)
   # Fit the model to the data
   # Note: Passing a single prior object will apply the same prior to all KCs in the data.
   #       To apply different priors to different KCs, pass a dict mapping KC ID to a prior object instead. 
   model.fit(data, priors=priors)

       

.. autosummary::
    :signatures: long
    :caption: Bayesian Priors
    :toctree: generated
   
    InitKnowledgeStrategy
    BayesianPriors

Fits (``stanbkt.fits``)
-----------------------
Module containing fit methods, fit configuration options, and fit result classes.

.. currentmodule:: stanbkt.fits


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

.. autosummary::
   :signatures: long
   :caption: Fit Configuration
   :toctree: generated
   
   BaseFitOptions
   MCMCFitOptions
   VBFitOptions
   MLEFitOptions
   PFFitOptions


Fit Results
^^^^^^^^^^^

Fit result objects for different inference methods.

These classes encapsulate the results of fitting a model to data, including parameter estimates, diagnostics, and other relevant information.
They are typically not used directly by users, however they can be used to access the underlying CmdStanPy Fit objects if needed.

.. autosummary::
   :signatures: long
   :caption: Fit Result Classes
   :toctree: generated
   

   BaseFit
   MCMCFit
   MLEFit
   VBFit
   PathfinderFit

Plotting (``stanbkt.plot``)
---------------------------

Posterior visualization of model correctness, including:
- Posterior predictive distributions of the probability of correctness
- Predictions sampled from the posterior predictive distribution

.. autosummary::
   :signatures: long
   :caption: Posterior Visualization Functions
   :toctree: generated
   
   plot_posterior_predictive_correctness

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

Logging Control
^^^^^^^^^^^^^^^

Control verbosity to specify the amount of logging information printed during model fitting and prediction.
Three levels are available: DEBUG, INFO, and WARN, listed in decreasing order of verbosity. 

.. code-block:: python
   :emphasize-lines: 2,5
  
   from stanbkt.utils import VerbosityLevel
   # Create Model with verbosity level set to WARNING
   model = StandardBKT(verbosity=VerbosityLevel.WARN)
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

