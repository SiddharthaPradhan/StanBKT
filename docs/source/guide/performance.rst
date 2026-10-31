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

- Each KC's fit is moved to the model's temporary folder after fitting and reloaded lazily when needed. Only one
  copy of the CSV files is kept.
- ``summary()`` and every ``predict*`` method release the fits they had to reload when they finish.
- Cached summaries never reload a fit.
- ``model.fits.get_fit(kc)`` reads the fit from disk on every call and the model does not keep it, so it is freed
  once you drop it. Hold on to the result if you need it more than once.
- Passing ``stan_output`` to ``predict_posterior_summary`` or ``predict_posterior_draws`` frees the cached draws of
  those objects afterwards. They are read from disk again if you need them.
- ``save()`` copies evicted fits from disk without loading them.

.. code-block:: python

   model = StandardBKT(low_memory=True, temp_dir="/scratch/stanbkt")
   model.fit(large_df)

``low_memory`` covers the fits. Stan posterior prediction still reads all draws of one KC into memory, which is
about 64 GB for 1 million observations and 4000 draws. Use ``backend="numba"`` if that is too much.

Temporary files
---------------

CmdStan writes its output to CSV files, which can be much larger than the data. The system temp dir is held in RAM
on many Linux machines, so these files can use up memory even though they are "on disk". The folder is chosen in
this order:

1. The ``temp_dir`` argument of the model or of :func:`~stanbkt.utils.model_io.load_model`.
2. The ``TMPDIR`` environment variable.
3. With ``low_memory=True``, ``tmp`` in the stanbkt user cache folder (``~/.cache/stanbkt/tmp`` on Linux).
4. Otherwise the system temp dir.

On Linux, if the chosen folder is held in RAM (``tmpfs``) and was not passed as ``temp_dir``, the model logs the
folder and its free space at the ``INFO`` level when it first needs temporary files.

Each model uses its own ``stanbkt_work_*`` subfolder and removes it on exit. Plan for this much free space:

- Fits: every chain of every KC, for as long as the model exists.
- ``save()``: a second copy of the fits while the archive is written.
- ``load_model``: the extracted archive, for as long as the model exists.
- Stan posterior prediction: the draws of the KC being processed, which ``predict_posterior_summary`` and
  ``predict_posterior_draws`` delete as they finish. The ``predict_posterior_stan`` output stays until you drop the
  returned objects, unless you moved it with ``save_csvfiles``.

A process that is killed cannot remove its folder. Leftover folders in the user cache are removed the next time
``stanbkt`` is imported, and those in any other location the next time a model creates a folder there. Folders of
running processes and the compiled model cache are never touched. On network drives without file lock support,
leftover folders are not removed automatically.

CmdStanPy's input data file stays in the system temp dir unless ``TMPDIR`` is set. It is small and is deleted after
each run.

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
