Saving and Loading
==================

A fitted model is saved to a single ``.stanbktmod`` archive and restored with :func:`~stanbkt.utils.model_io.load_model`.

.. code-block:: python

   from stanbkt import load_model

   model.save("models/skill_model")  # writes models/skill_model.stanbktmod
   model = load_model("models/skill_model")  # the suffix is optional

``save`` raises a ``RuntimeError`` if the model has no fits.

What is saved
-------------

- The CmdStan output files of every KC's fit.
- The cached parameter summaries, so ``summary()`` on a loaded model does not recompute them.
- Fit metadata: the fit method, summary percentiles, and per KC the group map (``group2index``), the student map
  (``student2index``), and the JOINT and post-test covariate columns.
- Model metadata: the model class and its constructor arguments (``fit_method``, ``verbose``,
  ``individual_initial_knowledge``, ``init_knowledge_strategy``, ``post_outcome_regression``,
  ``stan_compile_kwargs``, ``cpp_compile_kwargs`` and ``low_memory``).

``load_model`` rebuilds the same model class with the same arguments, so a loaded model can predict, summarize,
fit more KCs and be saved again. The Stan program is compiled (or taken from the cache) the first time it is
needed, not at load time.

Loading checks that the model and fit metadata agree on the fit method and raises a ``ValueError`` otherwise.

Low memory models
-----------------

A model created with ``low_memory=True`` is loaded lazily: each KC's fit is read from the archive only when a
method needs it, and released again afterwards. ``save`` copies fits that are already on disk without loading
them. See :doc:`performance`.

Compatibility
-------------

Archives store the information needed by the current version. Fits saved before per student and covariate maps
were persisted cannot predict with individualized initial knowledge and raise a ``ValueError`` asking you to
refit. Raw ``predict_posterior_stan`` output from versions before the flat ``pKnow`` and ``pCorrectness``
generated quantities is rejected by :func:`~stanbkt.utils.posterior_summary` and
:func:`~stanbkt.utils.gq_to_draws`, so rerun the prediction instead of reusing old Stan output.

See the :doc:`../examples/09_save_load_large_data` notebook for a worked example.
