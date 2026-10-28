import numpy as np
import pandas as pd
import pytest

import stanbkt.models.predictions as predictions_module
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils import posterior_summary
from stanbkt.utils.data_utils import ColumnNames, iter_kc_data
from stanbkt.utils.posterior_utils import (
    _flat_gq_slice,
    _observation_index_arrays,
)


def _sparse_df():
    # s1 has 2 interactions, s2 has 1, rows deliberately out of order
    return pd.DataFrame(
        {
            "student_id": ["s2", "s1", "s1"],
            "problem_id": ["p20", "p30", "p10"],
            "correct": [1, 0, 1],
            "kc_id": ["default_kc"] * 3,
            "timestamp": [1, 2, 1],
        }
    )


def _kc_data(df):
    mapping = ColumnNames.apply_default_mapping(None)
    return next(iter_kc_data(df, col_mapping=mapping, return_groups=False))[1]


class _FakeGQ:
    def __init__(self, draws: np.ndarray, column_names: tuple[str, ...], chains: int = 1):
        self._draws = draws
        self.column_names = column_names
        self.chains = chains

    def draws(self, *, concat_chains: bool = True) -> np.ndarray:
        return self._draws


class _FakeFit:
    def __init__(self, n_draws: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._params = {
            name: rng.uniform(low, high, size=n_draws)
            for name, (low, high) in {
                "pi_know": (0.1, 0.9),
                "learn": (0.05, 0.4),
                "forget": (0.0, 0.1),
                "guess": (0.05, 0.3),
                "slip": (0.05, 0.3),
            }.items()
        }

    def stan_variable(self, name: str) -> np.ndarray:
        return self._params[name]


def _numba_model(monkeypatch):
    model = StandardBKT()
    monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
    monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
    monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
    fit = _FakeFit()
    monkeypatch.setattr(model.fits, "get_fit", lambda kc_id: fit)
    return model


def test_observation_arrays_follow_student_then_interaction_order():
    students, problems, correctness, order = _observation_index_arrays(_kc_data(_sparse_df()))
    assert students.tolist() == ["s1", "s1", "s2"]
    assert problems.tolist() == ["p10", "p30", "p20"]
    assert correctness.tolist() == [1, 0, 1]
    assert order.tolist() == [0, 1, 0]


def test_flat_slice_locates_contiguous_columns():
    cols = ("lp__", "pKnow[1]", "pKnow[2]", "pKnow[3]", "pCorrectness[1]")
    assert _flat_gq_slice(cols, "pKnow", 3) == slice(1, 4)
    assert _flat_gq_slice(cols, "missing", 3) is None


@pytest.mark.parametrize(
    "cols",
    [
        # old padded layout
        ("pKnow[1,1]", "pKnow[1,2]", "pKnow[2,1]"),
        # generated from data with a different number of observations
        ("pKnow[1]", "pKnow[2]"),
        ("pKnow[1]", "pKnow[2]", "pKnow[3]", "pKnow[4]"),
    ],
)
def test_flat_slice_rejects_output_that_does_not_match_the_data(cols):
    with pytest.raises(ValueError, match="one value per observation|older"):
        _flat_gq_slice(cols, "pKnow", 3)


def test_mismatched_stan_output_raises_in_summary():
    gq = _FakeGQ(np.zeros((2, 2)), ("pKnow[1]", "pKnow[2]"))
    with pytest.raises(ValueError, match="one value per observation"):
        posterior_summary({"default_kc": gq}, data=_sparse_df())


def test_stan_output_draws_keep_chain_and_draw_ids():
    # 2 chains of 2 draws, values encode (draw, observation)
    values = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0], [40.0, 41.0, 42.0]])
    gq = _FakeGQ(values, ("pKnow[1]", "pKnow[2]", "pKnow[3]"), chains=2)
    model = StandardBKT()
    model._hidden_states_model = object()
    draws = model._process_predict_gq(
        {"default_kc": gq}, _sparse_df(), ColumnNames.apply_default_mapping(None)
    )["default_kc"]
    assert draws["chain__"].tolist() == [1] * 6 + [2] * 6
    assert draws["iter__"].tolist() == [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3
    assert draws["draw__"].tolist() == [d for d in (1, 2, 3, 4) for _ in range(3)]
    assert draws["pKnow"].tolist() == values.ravel().tolist()
    assert draws["student_id"].astype(str).tolist()[:3] == ["s1", "s1", "s2"]
    assert "pCorrectness" not in draws.columns


def test_draws_summary_quantiles_match_numpy():
    rng = np.random.default_rng(3)
    values = rng.uniform(size=(25, 3))
    gq = _FakeGQ(values, ("pKnow[1]", "pKnow[2]", "pKnow[3]"))
    model = StandardBKT()
    draws = model._process_predict_gq(
        {"default_kc": gq}, _sparse_df(), ColumnNames.apply_default_mapping(None)
    )
    quantiles = [0.05, 0.5, 0.975]
    out = posterior_summary(draws, quantiles=quantiles)
    for i in range(3):
        column = values[:, i]
        assert out["pKnow_mean"].iloc[i] == pytest.approx(column.mean())
        assert out["pKnow_std"].iloc[i] == pytest.approx(column.std(ddof=1))
        assert out["pKnow_median"].iloc[i] == pytest.approx(np.median(column))
        for q in quantiles:
            assert out[f"pKnow_{q * 100:.2f}%"].iloc[i] == pytest.approx(np.quantile(column, q))


@pytest.mark.parametrize("smoothed", [False, True])
def test_numba_summary_matches_summary_of_numba_draws(monkeypatch, smoothed):
    model = _numba_model(monkeypatch)
    data = _sparse_df()
    quantiles = [0.1, 0.9]
    draws_fn = model.predict_smoothed_posterior_draws if smoothed else model.predict_posterior_draws
    summary_fn = (
        model.predict_smoothed_posterior_summary if smoothed else model.predict_posterior_summary
    )
    expected = posterior_summary(draws_fn(data, backend="numba"), quantiles=quantiles)
    out = summary_fn(data, backend="numba", quantiles=quantiles)
    assert list(out.columns) == list(expected.columns)
    for col in out.columns:
        assert out[col].dtype == expected[col].dtype, col
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected.reset_index(drop=True), rtol=1e-12, atol=1e-12
    )


def test_numba_summary_does_not_build_draw_frames(monkeypatch):
    model = _numba_model(monkeypatch)

    def _fail(*args, **kwargs):
        raise AssertionError("draw-level frames should not be built for a summary")

    monkeypatch.setattr(predictions_module, "_predict_posterior_draws_numba", _fail)
    out = model.predict_posterior_summary(_sparse_df(), backend="numba")
    assert len(out) == 3
