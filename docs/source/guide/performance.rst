Memory and Performance
======================

Concurrent KC fitting
---------------------

KCs are independent fits, so ``fit`` runs several at once with ``n_kcs_workers`` (default ``-1``, automatic).
Results match a sequential fit.

- One fit already uses ``parallel_chains * threads_per_chain`` cores for MCMC, ``num_threads`` for Pathfinder and
  one core otherwise. The Stan settings take priority.
- With ``-1`` the number of workers is the available CPUs divided by the cores per fit, capped by the number of KCs.
- A positive value is used as given, but ``n_kcs_workers`` times the cores per fit must not exceed the available
  CPUs, otherwise a ``ValueError`` is raised.
- ``1`` always fits KCs one after another.

.. code-block:: python

   # 4 chains in parallel per KC, so on 16 cores about 4 KCs run at once
   model.fit(df, stan_fit_options=MCMCFitOptions(chains=4, parallel_chains=4))
   model.fit(df, n_kcs_workers=1, overwrite_kcs=True)  # sequential

Within chain threading
----------------------

For KCs with many students, split each chain's likelihood across threads. This requires compiling with
``STAN_THREADS``.

.. code-block:: python

   model = StandardBKT(cpp_compile_kwargs={"STAN_THREADS": True})
   model.fit(df, stan_fit_options=MCMCFitOptions(threads_per_chain=4))

The likelihood is split into slices of students. ``grainsize`` on every options class sets the number of students
per slice. The default ``-1`` picks it per KC: one slice when running on a single thread, otherwise about four
slices per thread. A positive value is used as given.

Threading helps when a KC has many students. With few students the threading overhead outweighs the gain, and
running more KCs concurrently is the better use of the cores.

GPU (OpenCL)
^^^^^^^^^^^^

Fit options accept ``opencl_ids`` for models compiled with ``STAN_OPENCL``. The BKT likelihood is a custom
recursion that OpenCL does not offload, so do not expect a speedup. Use ``STAN_THREADS`` instead.

Low memory mode
---------------

``low_memory=True`` keeps fits on disk instead of in memory, for models with many KCs.

- Each KC's fit is evicted to a temporary directory after fitting and reloaded lazily when needed.
- ``summary()`` and every ``predict*`` method release the fits they had to reload when they finish.
- Cached summaries never reload a fit.
- ``save()`` copies evicted fits from disk without loading them.

.. code-block:: python

   model = StandardBKT(low_memory=True)
   model.fit(large_df)

Large prediction outputs
------------------------

Draw level outputs have one row per posterior draw and interaction. ID columns are categorical and the
``chain__``, ``iter__``, ``draw__`` and ``_order`` columns use compact integer types, which brings a row to about
24 bytes, but 80 million rows is still about 2 GB. For large data:

- Use ``predict_posterior_summary`` (or the smoothed variant), which summarizes without keeping all draws.
- Use ``backend="numba"``, which avoids writing Stan CSV output.
- Keep ``n_cores`` at 1 unless you have memory to spare. Each concurrent KC holds its draws in memory.
- Use ``predict`` when parameter uncertainty is not needed.

Compilation cache
-----------------

The Stan program is compiled the first time a model is fitted and cached in the platform cache directory (for
example ``~/.cache/stanbkt/compiled_stan`` on Linux), keyed by the Stan source and compile options. Later models
with the same options reuse the binary, and a change to the Stan code triggers a rebuild. Each distinct
``cpp_compile_kwargs`` (for example with and without ``STAN_THREADS``) has its own binary.

.. code-block:: python

   from stanbkt.utils import list_cached_models, clear_stan_cache

   list_cached_models()
   clear_stan_cache()  # force a rebuild next time
