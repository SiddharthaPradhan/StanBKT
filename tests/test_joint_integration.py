"""Real CmdStan check of the JOINT strategy. Slow, excluded from the default run."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_options import MCMCFitOptions
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.utils.sim import sim_simple_BKT

pytestmark = pytest.mark.slow

KC = "kc_0"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def setup():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data = sim_simple_BKT(n_students=16, n_problems=5, n_kcs=1, rng_seed=7)
    ids = sorted(data["student_id"].unique())
    train_ids, new_ids = ids[:12], ids[12:]
    rng = np.random.default_rng(0)
    covariates = pd.DataFrame(
        {"student_id": ids, "pretest": rng.normal(size=len(ids))}
    )
    model = StandardBKT(
        fit_method=FitMethod.MCMC,
        individual_initial_knowledge=True,
        init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
    )
    options = MCMCFitOptions(
        chains=1, iter_warmup=100, iter_sampling=100, show_progress=False
    )
    model.fit(
        data[data["student_id"].isin(train_ids)],
        student_covariates=covariates,
        stan_fit_options=options,
    )
    return model, data, covariates, train_ids, new_ids


def _first_pknow(model, data, covariates):
    draws = model.predict_posterior_draws(data, student_covariates=covariates)[KC]
    draws = draws.assign(student_id=draws["student_id"].astype(str))
    return {
        sid: sub.groupby("draw__")["pKnow"].first().to_numpy()
        for sid, sub in draws.groupby("student_id")
    }


def _expected(model, covariates, sid, seen):
    fit = model.fits.get_fit(KC)
    b0 = fit.stan_variable("pi_b0_know_param").reshape(-1)
    b1 = fit.stan_variable("pi_b1_know_param").reshape(len(b0), -1)
    sigma = fit.stan_variable("pi_sigma_param").reshape(-1)
    z = fit.stan_variable("logit_pi_know_z")
    x = covariates.set_index("student_id").loc[sid, ["pretest"]].to_numpy()
    logit = b0 + b1 @ x
    if seen:
        index = model.fits.get_fit_save_entry(KC).student2index[sid]
        logit = logit + sigma * z[:, index - 1]
    return _sigmoid(logit)


def test_seen_students_reuse_fitted_z(setup):
    model, data, covariates, train_ids, _ = setup
    subset = train_ids[:4]
    first = _first_pknow(model, data[data["student_id"].isin(subset)], covariates)
    for sid in subset:
        np.testing.assert_allclose(
            first[sid], _expected(model, covariates, sid, seen=True), atol=1e-8
        )


def test_unseen_students_use_regression_mean(setup):
    model, data, covariates, _, new_ids = setup
    first = _first_pknow(model, data[data["student_id"].isin(new_ids)], covariates)
    for sid in new_ids:
        np.testing.assert_allclose(
            first[sid], _expected(model, covariates, sid, seen=False), atol=1e-8
        )


def test_results_unchanged_under_shuffled_rows(setup):
    model, data, covariates, train_ids, new_ids = setup
    mixed = data[data["student_id"].isin(train_ids[:3] + new_ids[:2])]
    shuffled = mixed.sample(frac=1.0, random_state=3)
    ordered = _first_pknow(model, mixed, covariates)
    reordered = _first_pknow(model, shuffled, covariates)
    assert ordered.keys() == reordered.keys()
    for sid in ordered:
        np.testing.assert_allclose(ordered[sid], reordered[sid], atol=1e-12)


def test_default_fit_has_no_joint_parameters_and_samples_cleanly():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data = sim_simple_BKT(n_students=30, n_problems=8, n_kcs=1, rng_seed=5)
    model = StandardBKT(fit_method=FitMethod.MCMC)
    model.fit(
        data,
        stan_fit_options=MCMCFitOptions(
            chains=2, iter_warmup=150, iter_sampling=150, seed=42, show_progress=False
        ),
    )
    fit = model.fits.get_fit(KC)
    joint_like = [
        c for c in fit.column_names if "pi_b0" in c or "pi_b1" in c or "pi_sigma" in c or "_z[" in c
    ]
    assert joint_like == []
    assert int(np.sum(fit.method_variables()["divergent__"])) == 0
    assert np.all(np.isfinite(model.summary()["Mean"].to_numpy()))
