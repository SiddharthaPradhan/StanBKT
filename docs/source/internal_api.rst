Internal API Reference
======================

This section documents the complete internal API for developers and contributors.
Full module paths are preserved for precise reference.

.. note::
   The internal API documented here is for developers and advanced users.
   **For normal usage, see the** :doc:`public_api` **instead.**

Package Structure
-----------------

.. toctree::
   :maxdepth: 4

   api/stanbkt


Detailed Module Documentation
------------------------------

Models
~~~~~~

.. toctree::
   :maxdepth: 3

   api/stanbkt.models
   api/stanbkt.models.core.base
   api/stanbkt.models.core.standard
   api/stanbkt.models.core.grouped
   api/stanbkt.models.core.nested
   api/stanbkt.models.priors
   api/stanbkt.models.model_types
   api/stanbkt.models.error

Fits
~~~~

.. toctree::
   :maxdepth: 3

   api/stanbkt.fits
   api/stanbkt.fits.core.base
   api/stanbkt.fits.core.mcmc
   api/stanbkt.fits.core.mle
   api/stanbkt.fits.core.vb
   api/stanbkt.fits.core.pf
   api/stanbkt.fits.fit_factory
   api/stanbkt.fits.fit_options
   api/stanbkt.fits.fit_types
   api/stanbkt.fits.persistence.fit_io
   api/stanbkt.fits.persistence.metadata

Utils
~~~~~

.. toctree::
   :maxdepth: 3

   api/stanbkt.utils
   api/stanbkt.utils.compilation
   api/stanbkt.utils.data_utils
   api/stanbkt.utils.verbose
