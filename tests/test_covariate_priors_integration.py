"""Real CmdStan checks of per covariate flat priors. Slow, excluded from the default run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.models.priors import MultiPriors, PriorsBase, StandardPriors
from stanbkt.utils.sim import sim_grouped_BKT, sim_simple_BKT
from stanbkt.utils.verbose import VerbosityLevel

pytestmark = pytest.mark.slow

_FIT_OPTIONS = {
    "chains": 2,
    "parallel_chains": 2,
    "iter_warmup": 300,
    "iter_sampling": 300,
    "seed": 3,
    "show_progress": False,
}
# the mean of the first prior is far from the data, so a tight posterior there shows it was applied
_MU = [5.0, None]
_STD = [0.01, 0.01]


def _covariates(student_ids) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"student_id": sorted(set(student_ids))})
    frame["a"] = rng.normal(size=len(frame))
    frame["b"] = rng.normal(size=len(frame))
    return frame


def _assert_first_held_and_second_flat(draws: np.ndarray) -> None:
    assert abs(draws[:, 0].mean() - 5.0) < 0.05
    assert draws[:, 0].std() < 0.05
    assert draws[:, 1].std() > 0.1


def test_joint_standard():
    data = sim_simple_BKT(n_students=60, n_problems=15, n_kcs=1, rng_seed=1)
    model = StandardBKT(
        individual_initial_knowledge=True,
        init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        verbose=VerbosityLevel.SILENT,
    )
    model.fit(
        data,
        student_covariates=_covariates(data["student_id"]),
        priors=StandardPriors(pi_b1_know_mu=_MU, pi_b1_know_std=_STD),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    _assert_first_held_and_second_flat(model.fits.get_fit(kc).stan_variable("pi_b1_know_param"))


def test_joint_multi():
    data = sim_grouped_BKT(n_students=60, n_problems=15, n_kcs=1, n_groups=2, rng_seed=1)
    model = MultiBKT(
        individual_initial_knowledge=True,
        init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        verbose=VerbosityLevel.SILENT,
    )
    model.fit(
        data,
        student_covariates=_covariates(data["student_id"]),
        priors=MultiPriors(pi_b1_know_mu=_MU, pi_b1_know_std=_STD),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    _assert_first_held_and_second_flat(model.fits.get_fit(kc).stan_variable("pi_b1_know_param"))


@pytest.mark.parametrize("grouped", [False, True], ids=["standard", "multi"])
def test_post_outcome_regression(grouped):
    if grouped:
        data = sim_grouped_BKT(n_students=60, n_problems=15, n_kcs=1, n_groups=2, rng_seed=1)
        model_class, priors_class = MultiBKT, MultiPriors
    else:
        data = sim_simple_BKT(n_students=60, n_problems=15, n_kcs=1, rng_seed=1)
        model_class, priors_class = StandardBKT, StandardPriors
    students = sorted(set(data["student_id"]))
    rng = np.random.default_rng(1)
    post_outcome = pd.DataFrame({"student_id": students, "post_outcome": rng.normal(size=len(students))})
    model = model_class(post_outcome_regression=True, verbose=VerbosityLevel.SILENT)
    model.fit(
        data,
        post_outcome=post_outcome,
        student_post_covariates=_covariates(students),
        priors=priors_class(post_b1_mu=_MU, post_b1_std=_STD),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    _assert_first_held_and_second_flat(model.fits.get_fit(kc).stan_variable("post_b1_param"))


def test_default_and_all_flat_priors_still_fit():
    data = sim_simple_BKT(n_students=40, n_problems=10, n_kcs=1, rng_seed=1)
    for priors in (None, StandardPriors(pi_b1_know_mu=None)):
        model = StandardBKT(
            individual_initial_knowledge=True,
            init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
            verbose=VerbosityLevel.SILENT,
        )
        model.fit(
            data,
            student_covariates=_covariates(data["student_id"]),
            priors=priors,
            stan_fit_options={**_FIT_OPTIONS, "iter_warmup": 100, "iter_sampling": 50},
        )
        assert model.fits.num_fitted_kcs == 1


@pytest.mark.parametrize(
    "flag, slope_near_50",
    [(1, False), (0, True)],
    ids=["flag_on_ignores_the_prior", "flag_off_control"],
)
def test_stan_only_applies_the_prior_when_the_flag_is_off(monkeypatch, flag, slope_near_50):
    # the second prior is impossible to miss, so the slope shows whether Stan applied it
    monkeypatch.setattr(
        PriorsBase,
        "_covariate_prior_data",
        staticmethod(lambda *args: ([0.0, 50.0], [5.0, 0.01], [0, flag])),
    )
    data = sim_simple_BKT(n_students=60, n_problems=15, n_kcs=1, rng_seed=1)
    model = StandardBKT(
        individual_initial_knowledge=True,
        init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        verbose=VerbosityLevel.SILENT,
    )
    model.fit(
        data,
        student_covariates=_covariates(data["student_id"]),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    slope = float(model.fits.get_fit(kc).stan_variable("pi_b1_know_param")[:, 1].mean())
    print(f"flag={flag} second slope mean={slope:.3f}")
    assert bool(abs(slope - 50.0) < 1.0) is slope_near_50, f"slope {slope}"


def _grouped_data():
    return sim_grouped_BKT(n_students=60, n_problems=15, n_kcs=1, n_groups=2, rng_seed=1)


def test_group_list_with_none_holds_only_the_group_with_a_prior():
    model = MultiBKT(verbose=VerbosityLevel.SILENT)
    model.fit(
        _grouped_data(),
        priors=MultiPriors(learn_mu=[1.0, None], learn_std=[0.01, 0.01]),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    draws = model.fits.get_fit(kc).stan_variable("logit_learn_group")
    assert abs(draws[:, 0].mean() - 1.0) < 0.05
    assert draws[:, 0].std() < 0.05
    assert draws[:, 1].std() > 0.1


@pytest.mark.parametrize(
    "flag, near_prior",
    [(1, False), (0, True)],
    ids=["flag_on_ignores_the_prior", "flag_off_control"],
)
def test_stan_applies_group_priors_only_when_the_flag_is_off(monkeypatch, flag, near_prior):
    model = MultiBKT(verbose=VerbosityLevel.SILENT)
    build = model._build_stan_data_dict

    def _with_tight_second_prior(*args, **kwargs):
        stan_data = build(*args, **kwargs)
        stan_data["prior_learn_mu"] = [0.0, 8.0]
        stan_data["prior_learn_std"] = [5.0, 0.01]
        stan_data["unif_prior_learn"] = [0, flag]
        return stan_data

    monkeypatch.setattr(model, "_build_stan_data_dict", _with_tight_second_prior)
    model.fit(_grouped_data(), stan_fit_options=_FIT_OPTIONS)
    kc = next(iter(model.fits.get_fitted_kcs()))
    second = float(model.fits.get_fit(kc).stan_variable("logit_learn_group")[:, 1].mean())
    print(f"group flag={flag} second group logit learn mean={second:.3f}")
    assert bool(abs(second - 8.0) < 1.0) is near_prior, f"mean {second}"


def test_individual_initial_knowledge_uses_the_flag_of_each_students_group():
    data = _grouped_data()
    model = MultiBKT(individual_initial_knowledge=True, verbose=VerbosityLevel.SILENT)
    model.fit(
        data,
        priors=MultiPriors(pi_know_mu=[3.0, None], pi_know_std=[0.01, 0.01]),
        stan_fit_options=_FIT_OPTIONS,
    )
    kc = next(iter(model.fits.get_fitted_kcs()))
    fit_save = model.fits.get_fit_save_entry(kc)
    draws = model.fits.get_fit(kc).stan_variable("logit_pi_know_group")
    groups = data.drop_duplicates("student_id").set_index("student_id")["group_id"]
    first_group = next(g for g, i in fit_save.group2index.items() if i == 1)
    students = sorted(fit_save.student2index, key=fit_save.student2index.get)
    in_first = np.array([str(groups[int(s)] if s.isdigit() else groups[s]) == str(first_group) for s in students])
    assert in_first.any() and (~in_first).any()
    assert np.abs(draws[:, in_first].mean(axis=0) - 3.0).max() < 0.05
    assert draws[:, ~in_first].std(axis=0).min() > 0.1


def test_standard_model_flags_still_fit():
    data = sim_simple_BKT(n_students=40, n_problems=10, n_kcs=1, rng_seed=1)
    model = StandardBKT(verbose=VerbosityLevel.SILENT)
    model.fit(
        data,
        priors=StandardPriors(learn_mu=None),
        stan_fit_options={**_FIT_OPTIONS, "iter_warmup": 100, "iter_sampling": 50},
    )
    assert model.fits.num_fitted_kcs == 1
