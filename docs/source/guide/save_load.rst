Saving and Loading
==================

A fitted model is saved to a single ``.stanbktmod`` archive and restored with :func:`~stanbkt.utils.model_io.load_model`.

.. code-block:: python

   from stanbkt import load_model

   model.save("models/skill_model")  # writes models/skill_model.stanbktmod
   model = load_model("models/skill_model")  # the suffix is optional

``save`` raises a ``RuntimeError`` if the model has no fits.

``load_model`` also accepts ``temp_dir``, the folder the archive is extracted to (see :doc:`performance`), and
``low_memory``, which overrides the saved setting.

What is saved
-------------

- The CmdStan output files of every KC's fit.
- The cached parameter summaries, so ``summary()`` on a loaded model does not recompute them.
- Fit metadata: the fit method, summary percentiles, and per KC the group map (``group2index``), the student map
  (``student2index``), and the JOINT and post-test covariate columns.
- Model metadata: the model class and its constructor arguments (``fit_method``, ``verbose``,
  ``individual_initial_knowledge``, ``init_knowledge_strategy``, ``post_outcome_regression``,
  ``guess_slip_scale``, ``stan_compile_kwargs``, ``cpp_compile_kwargs`` and ``low_memory``).

``load_model`` rebuilds the same model class with the same arguments, so a loaded model can predict, summarize,
fit more KCs and be saved again. The Stan program is compiled (or taken from the cache) the first time it is
needed, not at load time.

Loading checks that the model and fit metadata agree on the fit method and raises a ``ValueError`` otherwise.

Low memory models
-----------------

A model created with ``low_memory=True``, or loaded with ``load_model(path, low_memory=True)``, is loaded lazily: the
archive is extracted to disk, each KC's fit is read only when a method needs it, and released again afterwards.
``save`` copies fits that are already on disk without loading them. See :doc:`performance`.

Keeping generated quantities
----------------------------

Generated quantities are not part of the archive. They can be recomputed from the fit and the data, as the Stan
programs use no random numbers, so saving the prediction output is usually all you need:

.. code-block:: python

   model.predict_posterior_summary(data).to_csv("predictions.csv", index=False)

To keep the raw CmdStan output of ``predict_posterior_stan``, move it out of the temporary folder. Use one folder
per KC, and read the files back with pandas, as CmdStanPy has no function for loading them:

.. code-block:: python

   import glob
   import pandas as pd

   gq = model.predict_posterior_stan(data)
   for kc, gq_fit in gq.items():
       gq_fit.save_csvfiles(f"predictions/{kc}")

   chains = [pd.read_csv(path, comment="#") for path in sorted(glob.glob("predictions/kc_1/*.csv"))]

Compatibility
-------------

Archives store the information needed by the current version. Fits saved before per student and covariate maps
were persisted cannot predict with individualized initial knowledge and raise a ``ValueError`` asking you to
refit. Raw ``predict_posterior_stan`` output from versions before the flat ``pKnow`` and ``pCorrectness``
generated quantities is rejected by :func:`~stanbkt.utils.posterior_summary` and
:func:`~stanbkt.utils.gq_to_draws`, so rerun the prediction instead of reusing old Stan output.

See the :doc:`../examples/09_save_load_large_data` notebook for a worked example.
