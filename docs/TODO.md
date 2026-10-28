# Documentation TODO

Backlog of concepts, API pages, and examples that the ReadTheDocs site (docs/source) still needs. Remove an item when it is done. Add an item whenever a change ships without its docs (see CLAUDE.md).

Last full scan: 2026-09-18.

## 1. Broken or stale existing docs (fix first)

- [ ] `docs/source/index.rst` toctree lists `guide`, but no `guide.rst` exists. Either create the User's Guide (section 2) or remove the entry. `quickstart.rst` is also not in any toctree.
- [ ] `quickstart.rst`: says Python 3.11+ (package requires 3.12+); contains a leftover TODO (Windows note) and FIXME (ASSISTments mapping); has a broken nested `.. code-block::` inside the column mapping example.
- [ ] `public_api.rst` Bayesian Priors example uses `StandardPriors(pi_know_mu=..., pi_know_sigma=...)`. The real field is `pi_know_std`.
- [ ] `public_api.rst` Logging Control example uses `StandardBKT(verbosity=...)`. The real argument is `verbose`.
- [ ] `public_api.rst` InitKnowledgeStrategy example describes JOINT as using "a pretest". JOINT now takes any number of student level covariates; the example must show `student_covariates` (see section 2).
- [ ] `public_api.rst` and `docs/source/generated/` are missing `MultiPriors`, `FitMetadata`, `FitFactory`, `StanFitOptions`, `ModelType`, `FitMethodMismatchError`, and the fit classes are only listed, not explained.
- [ ] `docs/source/generated/` contains stale autosummary stubs; regenerate after the public API page is updated.
- [ ] Top level `stanbkt/__init__.py` does not export `MultiBKT`, `MultiPriors`, or `InitKnowledgeStrategy`, while the docs import them from `stanbkt.models`. Decide the canonical import path and document it.
- [ ] `example_root.rst` cards describe the Multi-BKT notebook as "multiple BKT models and multi-skill scenarios". It actually covers grouped parameters; fix the wording.
- [ ] Untracked duplicate notebooks in `docs/source/examples/` (`02_simple_example copy 2.ipynb`, `02_simple_example_copy.ipynb`, `02_simple_example_sim.ipynb`, the `.eps` files) are scratch, not part of the gallery. Delete or move them out of `docs/`.
- [ ] `docs/build/` is checked in build output; confirm it is gitignored and not hand edited.

## 2. Concepts and guide pages to write

Create `docs/source/guide/` (or a single `guide.rst`) with these pages:

- [ ] **BKT model overview**: the four parameters (`pi_know`, `learn`, `forget`, `guess`, `slip` and their constraints, guess and slip capped at 0.5), the hidden Markov formulation, what `pKnow` and `pCorrectness` mean.
- [ ] **Data format**: long format columns, `column_mapping`, ordering column, optional `kc_id`, how KCs are split and fitted independently, how missing or ragged sequences are padded.
- [ ] **Choosing a model**: `StandardBKT` versus `MultiBKT` (group column, group2index alignment at predict time), when to use each.
- [ ] **Priors**: logit scale parameterization, `StandardPriors`, `MultiPriors` (scalar or per group lists), uniform (non informative) priors via `None`, per KC prior dicts, defaults, `use_defaults`.
- [ ] **Inference methods**: MCMC, variational, MLE, Pathfinder; fit options classes; which prediction paths each supports (`generate_quantities` rejects Pathfinder fits); threading (`STAN_THREADS` and `threads_per_chain`).
- [ ] **Initial knowledge**: population level, `individual_initial_knowledge=True` with `CORRECTNESS_ONLY` (one free `pi_know` per student; unseen students at predict time are dropped with a warning, error if none remain), and `JOINT` (regression of logit `pi_know` on student covariates with non centered residual `z`).
- [ ] **JOINT strategy in depth**: model equation `logit(pi_know) = b0 + X b1 + sigma * z`, the `student_covariates` DataFrame (one row per student, covariate columns inferred from the frame, column order fixed at fit time), duplicate ID warning, errors for a missing student, non numeric columns, NaN or infinite values, and a frame with no covariate columns, prior fields (`pi_b0_know_*`, `pi_b1_know_*` scalar or per covariate list, `pi_sigma_lambda` as an Exponential rate), seen versus unseen student behavior at inference (seen reuse fitted `z`, unseen use the regression mean), how to read the `pi_b0_know_param`, `pi_b1_know_param` (one row per covariate), `pi_sigma_param`, `logit_pi_know_z` rows in `summary()` (these rows exist only under JOINT; default and `CORRECTNESS_ONLY` fits have exactly the same parameters as before).
- [ ] **Post-test outcome regression in depth** (new, `post_outcome_regression=True`): model equation `post_outcome ~ normal(post_b0 + post_b_mastery * final_pknow + post_covariates * post_b1, post_sigma)`, where `final_pknow` is each student's filtered (forward-pass) mastery probability after their last observed interaction, computed jointly with the BKT fit (not post-hoc). Independent of `init_knowledge_strategy`/`individual_initial_knowledge` and can be combined with JOINT. Document `post_outcome`/`student_post_covariates` DataFrame shapes (mirrors `student_covariates`, but `post_outcome` requires exactly one outcome column and covariates are optional), the `post_b0_param`/`post_b_mastery_param`/`post_b1_param`/`post_sigma_param` summary rows, and that `predict_post_outcome*` prediction methods are not yet implemented (see API gaps below).
- [ ] **Prediction guide**: point estimate (`predict`, `predict_smoothed`, mean/median/mode, numba `parallel` and `fast_math`) versus posterior (`predict_posterior_stan`, `_draws`, `_summary`, smoothed variants, `backend="stan"` versus `"numba"`, `n_cores` and its memory cost). Filtered (online) versus smoothed (offline) states.
- [ ] **Summaries and diagnostics**: `model.summary()`, percentiles, summary cache, `posterior_summary`, `summarize_draws`, `summary_parameter_names`, how to interpret R-hat and ESS for MCMC fits. Also document the labeled summary index `[kc, parameter, axis, label]` and `label_indexes=False` for raw CmdStan names, and update any example notebook that indexes summary rows by `learn[1]` style names.
- [ ] **Saving and loading**: `model.save`, `load_model`, what is persisted (fit CSVs, summary cache, metadata including group and student index maps and covariate schema), version compatibility of old archives.
- [ ] **Memory and performance**: `low_memory=True` (evicts fits to disk after each KC, lazily reloads, and `summary()` drops fits it had to reload), categorical ID columns and compact integer dtypes (`draw__`, `chain__`, `iter__`, `_order`) in draw-level outputs (about 24 bytes per row, so 80M rows is roughly 2 GB), using `_summary` prediction variants for large data, `n_cores` trade offs for prediction, `fit(n_kcs_workers=-1)` (concurrent KC fitting, on by default in auto mode: the Stan settings take priority, so workers = available CPUs divided by the cores one fit already uses, `parallel_chains * threads_per_chain` for MCMC, capped by the number of KCs; an explicit value raises a `ValueError` if `n_kcs_workers * cores per fit` exceeds the available CPUs, and `1` is always sequential), compilation cache.
- [ ] **Evaluation metrics**: `accuracy`, `rmse`, `auc` and how to apply them per KC.
- [ ] **Troubleshooting / FAQ**: CmdStan not installed, `setup_cmdstanpy` raising on a GitHub rate limit (set `GITHUB_PAT`; the installation page has a short note), compiler issues on Windows, `STAN_THREADS` mismatch error, Pathfinder with posterior prediction, unseen student warnings, duplicate covariate IDs.

## 3. API reference gaps (`public_api.rst`)

Undocumented public objects, with the section each belongs in:

- [ ] `stanbkt.models`: `MultiPriors`, `ModelType`, `FitMethodMismatchError`, `predict_posterior`.
- [ ] `stanbkt.fits`: `StanFitOptions`, `FitMetadata`, `FitFactory`.
- [ ] `stanbkt.plot`: `plot_dist`, `plot_trace` (parameter plots; only `plot_posterior_correctness` is listed).
- [ ] `stanbkt.utils`: `load_model`, `sim_simple_BKT`, `sim_grouped_BKT`, `accuracy`, `rmse`, `auc`, `posterior_summary`, `gq_to_draws`, `summarize_draws`, `summary_parameter_names`, `KCData`, `validate_data`, `compile_stan_model`.
- [ ] Model methods worth their own subsection with examples: `fit` (including `overwrite_kcs`, `student_covariates`, `post_outcome`, `student_post_covariates`, `n_kcs_workers`), `predict`, `predict_smoothed`, all `predict_posterior*` and `predict_smoothed_posterior*`, `summary`, `save`, `set_verbosity`.
- [ ] Constructor options not yet described anywhere: `low_memory`, `individual_initial_knowledge`, `init_knowledge_strategy` (also on `MultiBKT`), `post_outcome_regression` (also on `MultiBKT`), `stan_compile_kwargs`, `cpp_compile_kwargs`.
- [ ] Docstring audit: every new public parameter needs a numpydoc entry (recent additions: `student_covariates`, `post_outcome`, `student_post_covariates`, `low_memory`, `lazy` loading).
- [ ] **`predict_post_outcome`/`predict_post_outcome_posterior` not yet implemented.** `post_outcome_regression=True` currently only supports the joint fit-time estimation (`post_b0_param`/`post_b_mastery_param`/`post_b1_param`/`post_sigma_param` are estimated and readable via `summary()`); there is no scoring path yet that combines a held-out student's predicted final mastery with these coefficients to produce a predicted post-test outcome. Since final mastery here is a deterministic function of fitted BKT parameters and each student's own history (not a free per-student latent like JOINT's `z`), no seen/unseen split should be needed, unlike the JOINT prediction path. `StandardPriors`/`MultiPriors` and the Stan data dict already carry `post_b0_mu/std`, `post_b_mastery_mu/std`, `post_b1_mu/std`, `post_sigma_lambda`; add the prediction methods to `models/predictions.py`/`models/core/base.py` following the `predict`/`predict_posterior` split, and thread `post_outcome`/`student_post_covariates` through `_prepare_post_outcome_data` at predict time (only threaded through `fit()` today).
- [ ] `student_post_covariates`/`post_outcome` are not yet threaded through `predict`/`predict_smoothed`/`predict_posterior*`/`predict_smoothed_posterior*` the way `student_covariates` is for JOINT (needed once `predict_post_outcome*` exists, to validate covariate columns match fit time via a `_check_post_covariate_columns_match` helper analogous to `_check_covariate_columns_match`).

## 4. Examples and notebooks to add

Existing: `01_setup`, `02_simple_example`, `03_multi_BKT_example`. Add, in the gallery order:

- [ ] **04 Priors**: default versus informative versus uniform priors, per KC prior dicts, `MultiPriors` per group lists, effect on posteriors.
- [ ] **05 Inference methods**: MCMC versus VB versus MLE versus Pathfinder on the same data, timing and accuracy, fit options.
- [ ] **06 Individualized initial knowledge**: `individual_initial_knowledge=True` with `CORRECTNESS_ONLY`, including the unseen student drop behavior.
- [ ] **07 JOINT initial knowledge with covariates**: simulate data with 1 and several covariates, fit, read `b0/b1/sigma` from `summary()`, predict for seen and unseen students, compare with the no covariate model.
- [ ] **07b Post-test outcome regression**: simulate data where a post-test outcome depends on final mastery (and optionally post-test covariates), fit with `post_outcome_regression=True`, read `post_b0_param`/`post_b_mastery_param`/`post_b1_param`/`post_sigma_param` from `summary()`, and combine with `InitKnowledgeStrategy.JOINT` in the same fit. Add once `predict_post_outcome*` exists (see API gaps).
- [ ] **08 Multiple KCs and real data**: fitting many KCs, `column_mapping` on a public dataset (ASSISTments style), per KC prior dict, `overwrite_kcs`.
- [ ] **09 Saving, loading, and large data**: `save`/`load_model`, `low_memory=True`, streaming `_summary` predictions with `n_cores`.
- [ ] **10 Diagnostics and plotting**: `plot_trace`, `plot_dist`, `plot_posterior_correctness` (grouped and ungrouped), interpreting summaries.
- [ ] **11 Evaluation**: `accuracy`, `rmse`, `auc` on held out students with point estimate and posterior predictions.
- [ ] Add each new notebook to `example_root.rst` (card plus hidden toctree entry).
- [ ] Re-run all notebooks against the current release so outputs match the API (categorical ID columns, new summary rows for JOINT).

## 5. Build and site housekeeping

- [ ] Add a Changelog page (`docs/source/changelog.rst`) and link it from `index.rst`; record the memory work (including compact dtypes and the constant categorical KC column in draw-level prediction output), JOINT strategy, `low_memory`, and the `predict_posterior_summary` fix for pandas 2.x (categorical IDs produced extra all-NaN rows) as they ship.
- [ ] Add a Contributing/Development page (uv commands, running tests, slow tests marker, docs build, Stan code layout, release process: bump `pyproject.toml`, push a `v<version>` or `<version>` tag, `release.yml` runs CI then publishes to PyPI and creates a GitHub Release) sourced from CLAUDE.md.
- [ ] Decide whether nbsphinx should execute notebooks on build (`nbsphinx_execute`) or keep stored outputs; document the choice.
- [ ] Set `fail_on_warning` to true in `.readthedocs.yaml` once the stale references above are fixed.
- [ ] Draw-level prediction output (`predict_*posterior_draws`): optional `dtype=np.float32` for `pKnow` and `pCorrectness` (halves the float columns), student chunking for the Stan backend (`draws_pd()` is wide, peak is about 2.5x the final frame), and a possible wide (draw by observation) `output_format`. Not implemented; the full itv data (80M rows) now peaks at about 2 GB (numba) and 5.5 GB (stan).
- [ ] **GPU (OpenCL) example**: a short guide showing `STAN_OPENCL` with `opencl_ids`, expected speedups, and how to find platform/device ids. Untested on real GPU hardware so far.
