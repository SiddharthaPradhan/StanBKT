"""Real CmdStan check of the post-test outcome regression. Slow, excluded from the default run."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.model_types import InitKnowledgeStrategy

pytestmark = pytest.mark.slow


def _simulate(n_students=150, n_problems=15, seed=0):
    rng = np.random.default_rng(seed)
    true_pi, true_learn, true_guess, true_slip, true_forget = 0.3, 0.2, 0.2, 0.1, 0.02
    rows = []
    final_know_true = np.zeros(n_students)
    for s in range(n_students):
        known = rng.random() < true_pi
        for t in range(n_problems):
            p_correct = (1 - true_slip) if known else true_guess
            correct = int(rng.random() < p_correct)
            rows.append({"student_id": s, "problem_id": t, "correct": correct, "timestamp": t})
            known = (rng.random() < true_learn) if not known else (rng.random() > true_forget)
        final_know_true[s] = float(known)
    data = pd.DataFrame(rows)
    post = 2.0 + 5.0 * final_know_true + rng.normal(0, 0.5, n_students)
    post_outcome = pd.DataFrame({"student_id": range(n_students), "post_outcome": post})
    return data, post_outcome


def test_post_outcome_regression_recovers_positive_mastery_effect():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data, post_outcome = _simulate()
    model = StandardBKT(fit_method=FitMethod.MLE, post_outcome_regression=True)
    model.fit(data, post_outcome=post_outcome, stan_fit_options={"show_console": False})
    summary = model.summary()
    b_mastery = summary.xs("post_b_mastery_param", level="parameter")["value"].iloc[0]
    # true effect is 5.0; a positive, non-trivial recovered effect is enough to confirm
    # the joint fit is actually learning the mastery -> outcome relationship.
    assert b_mastery > 1.0


def test_default_fit_has_no_post_outcome_parameters():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data, _ = _simulate(n_students=20, n_problems=5)
    model = StandardBKT(fit_method=FitMethod.MLE)
    model.fit(data, stan_fit_options={"show_console": False})
    summary = model.summary()
    params = summary.index.get_level_values("parameter").unique()
    assert "post_b_mastery_param" not in params


def test_combined_with_joint_init_knowledge():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data, post_outcome = _simulate(n_students=80, n_problems=12, seed=1)
    ids = sorted(data["student_id"].unique())
    rng = np.random.default_rng(2)
    pretest = pd.DataFrame(
        {"student_id": ids, "pretest": rng.normal(size=len(ids))}
    )
    model = StandardBKT(
        fit_method=FitMethod.MLE,
        individual_initial_knowledge=True,
        init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        post_outcome_regression=True,
    )
    model.fit(
        data,
        student_covariates=pretest,
        post_outcome=post_outcome,
        stan_fit_options={"show_console": False},
    )
    summary = model.summary()
    params = summary.index.get_level_values("parameter").unique()
    assert "pi_b1_know_param" in params
    assert "post_b_mastery_param" in params


_FINAL_MASTERY_HARNESS = """
functions {
#include "final_mastery.stan"
}
data {
    int<lower=1> n;
    int<lower=1> T;
    array[n] int<lower=1> lengths;
    array[n, T] int<lower=-1, upper=1> correctness;
    vector[n] learn;
    vector[n] forget;
    vector[n] guess;
    vector[n] slip;
    vector[n] pi_know;
}
generated quantities {
    vector[n] final_pknow = final_pknow_forward(lengths, correctness, learn, forget, guess, slip, pi_know);
}
"""


def _final_pknow_reference(responses, pi_know, learn, forget, guess, slip):
    # filter on every response, then one more transition: P(know) at T+1
    k = pi_know
    for y in responses:
        num = k * (1 - slip) if y == 1 else k * slip
        denom = num + (1 - k) * (guess if y == 1 else 1 - guess)
        k = learn + (num / denom) * (1 - forget - learn)
    return k


def test_final_pknow_uses_last_response_and_projects_to_t_plus_one(tmp_path):
    from importlib.resources import files

    import cmdstanpy

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    stan_dir = str(files("stanbkt").joinpath("stan_code", "BKT"))
    stan_file = tmp_path / "final_mastery_harness.stan"
    stan_file.write_text(_FINAL_MASTERY_HARNESS)
    harness = cmdstanpy.CmdStanModel(
        stan_file=str(stan_file), stanc_options={"include-paths": [stan_dir]}
    )

    # the two students differ only in their last response, so the last response must matter
    sequences = [[0, 1, 1], [0, 1, 0], [1]]
    params = dict(pi_know=0.3, learn=0.2, forget=0.05, guess=0.2, slip=0.1)
    T = max(len(s) for s in sequences)
    correctness = [s + [-1] * (T - len(s)) for s in sequences]
    n = len(sequences)
    gq = harness.sample(
        data={
            "n": n,
            "T": T,
            "lengths": [len(s) for s in sequences],
            "correctness": correctness,
            **{key: [value] * n for key, value in params.items()},
        },
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        show_progress=False,
    )
    result = gq.stan_variable("final_pknow")[0]
    expected = [_final_pknow_reference(s, **params) for s in sequences]
    # CmdStan writes CSV output with 6 significant digits
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert result[0] > result[1]
