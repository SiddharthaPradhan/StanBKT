Public API Reference
====================

This page documents the public API that users should interact with.
All classes and functions listed here are stable and intended for public use.

Click on any item to see its detailed documentation.

Models
------

Core BKT model implementations.

.. currentmodule:: stanbkt.models

.. autosummary::
   :toctree: generated
   
   BKTModelBase
   StandardBKT

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   
   ModelType
   PriorEstimationType
   BayesianPriors

Exceptions
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   
   FitMethodMismatchError


Fits
----

Fit result objects for different inference methods.

.. currentmodule:: stanbkt.fits

.. autosummary::
   :toctree: generated
   
   BaseFit
   MCMCFit
   MLEFit
   VBFit
   PathfinderFit


Utilities
---------

Compilation and cache management utilities.

.. currentmodule:: stanbkt.utils

.. autosummary::
   :toctree: generated
   
   compile_stan_model
   get_stan_model_cache_dir
   clear_stan_cache
   get_cache_root
   list_cached_models
