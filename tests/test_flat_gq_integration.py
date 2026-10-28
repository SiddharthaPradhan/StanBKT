"""Real CmdStan checks of the flat generated quantities and the grainsize data. Slow, excluded from the default run."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import ColumnNames, iter_kc_data
from stanbkt.utils.sim import sim_grouped_BKT, sim_simple_BKT

pytestmark = pytest.mark.slow

_FIT_OPTIONS = {
    "chains": 2,
    "parallel_chains": 2,
    "iter_warmup": 100,
    "iter_sampling": 50,
    "seed": 3,
    "show_progress": False,
}
_PARAMS = [
    "logit_pi_know_group",
    "logit_learn_group",
    "logit_forget_group",
    "logit_guess_group",
    "logit_slip_group",
]


@pytest.fixture(scope="module", params=["standard", "multi"])
def fitted(request):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    # frac < 1 drops interactions so the sequence lengths differ between students
    if request.param == "standard":
        data = sim_simple_BKT(n_students=30, n_problems=10, n_kcs=2, rng_seed=1, frac=0.7)
        model = StandardBKT()
    else:
        data = sim_grouped_BKT(n_students=30, n_problems=10, n_kcs=1, n_groups=2, rng_seed=2, frac=0.7)
        model = MultiBKT()
    model.fit(data, stan_fit_options=_FIT_OPTIONS, n_kcs_workers=1)
    return model, data


@pytest.mark.parametrize("smoothed", [False, True])
def test_stan_draws_match_numba_draws(fitted, smoothed):
    model, data = fitted
    draws_fn = model.predict_smoothed_posterior_draws if smoothed else model.predict_posterior_draws
    stan_draws = draws_fn(data, backend="stan")
    numba_draws = draws_fn(data, backend="numba")
    assert set(stan_draws) == set(numba_draws)
    for kc, stan_df in stan_draws.items():
        numba_df = numba_draws[kc]
        assert len(stan_df) == len(numba_df)
        for col in ("student_id", "problem_id", "correct", "_order"):
            assert stan_df[col].astype(str).tolist() == numba_df[col].astype(str).tolist()
        # Stan writes 6 significant digits to its CSV output
        for col in ("pKnow", "pCorrectness"):
            np.testing.assert_allclose(stan_df[col], numba_df[col], rtol=1e-4, atol=1e-6)


def test_stan_summary_matches_numba_summary(fitted):
    model, data = fitted
    stan_summary = model.predict_posterior_summary(data, backend="stan")
    numba_summary = model.predict_posterior_summary(data, backend="numba")
    assert list(stan_summary.columns) == list(numba_summary.columns)
    for col in stan_summary.columns:
        if stan_summary[col].dtype.kind == "f":
            np.testing.assert_allclose(stan_summary[col], numba_summary[col], rtol=1e-4, atol=1e-6)
        else:
            assert stan_summary[col].astype(str).tolist() == numba_summary[col].astype(str).tolist()


def test_log_density_does_not_depend_on_grainsize(fitted, monkeypatch):
    model, data = fitted
    monkeypatch.setenv("STAN_NUM_THREADS", "4")
    threaded = compile_stan_model(
        model._stan_model_filename, cpp_options={"STAN_THREADS": True}
    )
    mapping = ColumnNames.apply_default_mapping(None)
    for kc_id, kc_data in iter_kc_data(data, col_mapping=mapping, return_groups=model._use_groups):
        kc_data = model._align_kc_group_indices_with_fit_metadata(str(kc_id), kc_data)
        data_dict = model._build_stan_data_dict(kc_data, model._default_priors())
        fit = model.fits.get_fit(str(kc_id))
        params = {name: np.asarray(fit.stan_variable(name))[7].tolist() for name in _PARAMS}
        results = []
        for grainsize in (1, 2, 5, data_dict["nStudents"]):
            lp = threaded.log_prob(
                params=params, data={**data_dict, "grainsize": grainsize}, sig_figs=18
            )
            results.append(lp.to_numpy().ravel())
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], rtol=1e-12, atol=1e-12)
