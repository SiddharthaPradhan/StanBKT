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
