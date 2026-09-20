"""Tests for the JOINT init-knowledge strategy (mock based, no CmdStan)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import stanbkt.fits.core.base as fit_base_module
import stanbkt.models.predictions as predictions_module
from stanbkt.fits.fit_types import FitMetadata, FitMethod, FitSaveEntry
from stanbkt.fits.persistence.metadata import (
    fit_metadata_from_json,
    fit_metadata_to_json,
)
from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.models.predictions import _prepare_posterior_prediction_inputs
from stanbkt.utils.data_utils import (
    ColumnNames,
    KCData,
    StudentInteraction,
    format_kc_data,
)

JOINT = InitKnowledgeStrategy.JOINT


def _df(students=("s1", "s2", "s3")) -> pd.DataFrame:
    rows = []
    for s in students:
        for t, (p, c) in enumerate([("p1", 1), ("p2", 0), ("p3", 1)], start=1):
            rows.append(
                {"student_id": s, "problem_id": p, "correct": c, "timestamp": t}
            )
    return pd.DataFrame(rows)


def _covariates(students=("s1", "s2", "s3")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": list(students),
            "pretest": np.linspace(-1, 1, len(students)),
            "age": np.linspace(10, 12, len(students)),
        }
    )


def _joint_model(cls=StandardBKT, **kwargs):
    return cls(
        individual_initial_knowledge=True, init_knowledge_strategy=JOINT, **kwargs
    )


def _mock_fit(model, monkeypatch):
    monkeypatch.setattr(
        model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
    )
    captured: list[dict] = []

    def _fake_fit(data_dict, fit_options):
        captured.append(data_dict)
        return MagicMock()

    monkeypatch.setattr(model, "_fit_stan_model_using_method", _fake_fit)
    monkeypatch.setattr(
        FitMethod,
        "infer_fit_method_from_stan_fit",
        staticmethod(lambda _: FitMethod.MCMC),
    )
    return captured


class TestFitValidation:
    def test_fit_raises_without_covariates(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="student_covariates"):
            model.fit(_df())

    def test_fit_raises_for_missing_student(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="s3"):
            model.fit(_df(), student_covariates=_covariates(("s1", "s2")))

    def test_fit_warns_and_dedups_duplicate_ids(self, monkeypatch):
        model = _joint_model()
        captured = _mock_fit(model, monkeypatch)
        dup = pd.concat([_covariates(), _covariates(("s1",))], ignore_index=True)
        with pytest.warns(UserWarning, match="duplicate student IDs"):
            model.fit(_df(), student_covariates=dup)
        assert captured[0]["covariates"].shape == (3, 2)

    def test_non_joint_model_ignores_covariates(self, monkeypatch):
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        model.fit(_df())
        assert captured[0]["joint_pi_know"] == 0
        assert captured[0]["nCovariates"] == 0


class TestFitPersistence:
    def test_covariate_columns_persisted(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), student_covariates=_covariates())

        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.covariate_columns == ("pretest", "age")
        assert entry.student2index == {"s1": 1, "s2": 2, "s3": 3}

    def test_non_joint_fit_persists_no_covariates(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.covariate_columns is None
        assert entry.student2index is None

    def test_correctness_only_individual_persists_student2index(self, monkeypatch):
        model = StandardBKT(individual_initial_knowledge=True)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"s1": 1, "s2": 2, "s3": 3}
        assert entry.covariate_columns is None

    @pytest.mark.parametrize("n_cov", [1, 2])
    def test_stan_data_carries_arbitrary_covariates(self, monkeypatch, n_cov):
        model = _joint_model()
        captured = _mock_fit(model, monkeypatch)
        cov = _covariates()[["student_id", "pretest", "age"][: n_cov + 1]]
        model.fit(_df(), student_covariates=cov)
        data = captured[0]
        assert data["joint_pi_know"] == 1
        assert data["nCovariates"] == n_cov
        assert len(data["prior_pi_b1_know_mu"]) == n_cov
        assert data["unif_prior_pi_know"] == 1
        assert data["unif_prior_pi_b0_know"] == 0


class TestMetadataRoundTrip:
    def test_covariate_columns_round_trip(self):
        entry = FitSaveEntry(
            kc="kc_a",
            save_folder="kc_a_12345678",
            covariate_columns=("pretest", "age"),
        )
        metadata = FitMetadata(fit_method=FitMethod.MCMC, fit_saves={"kc_a": entry})
        parsed = fit_metadata_from_json(fit_metadata_to_json(metadata))
        assert parsed.fit_saves["kc_a"] == entry

    def test_old_metadata_without_covariate_columns_loads(self):
        raw = (
            '{"fit_method": "mcmc", "fit_saves": [{"kc": "k", "save_folder": "f"}],'
            ' "summary_percentiles": [2.5, 97.5]}'
        )
        entry = fit_metadata_from_json(raw).fit_saves["k"]
        assert entry.covariate_columns is None

    def test_student2index_round_trip(self):
        entry = FitSaveEntry(
            kc="kc_a",
            save_folder="kc_a_12345678",
            student2index={"s1": 1, "s2": 2},
            covariate_columns=("pretest",),
        )
        metadata = FitMetadata(fit_method=FitMethod.MCMC, fit_saves={"kc_a": entry})
        parsed = fit_metadata_from_json(fit_metadata_to_json(metadata))
        assert parsed.fit_saves["kc_a"].student2index == {"s1": 1, "s2": 2}
        assert parsed.fit_saves["kc_a"] == entry

    def test_old_metadata_without_student2index_loads(self):
        raw = (
            '{"fit_method": "mcmc", "fit_saves": [{"kc": "k", "save_folder": "f",'
            ' "covariates_available": true}], "summary_percentiles": [2.5, 97.5]}'
        )
        entry = fit_metadata_from_json(raw).fit_saves["k"]
        assert entry.student2index is None
        assert entry.covariate_columns is None

    def test_hash_includes_covariate_columns(self):
        a = FitSaveEntry(kc="k", save_folder="f", covariate_columns=("a",))
        b = FitSaveEntry(kc="k", save_folder="f", covariate_columns=("b",))
        assert hash(a) != hash(b)

    def test_hash_includes_student2index(self):
        a = FitSaveEntry(kc="k", save_folder="f", student2index={"s1": 1})
        b = FitSaveEntry(kc="k", save_folder="f", student2index={"s1": 2})
        assert hash(a) != hash(b)


class TestMultiBKTConstructor:
    def test_accepts_and_stores_new_args(self):
        model = _joint_model(MultiBKT)
        assert model.individual_initial_knowledge is True
        assert model.init_knowledge_strategy == JOINT

    def test_joint_requires_individual_initial_knowledge(self):
        with pytest.raises(ValueError, match="individual_initial_knowledge"):
            MultiBKT(init_knowledge_strategy=JOINT)

    def test_init_kwargs_round_trip_new_args(self):
        kwargs = _joint_model(MultiBKT)._get_model_init_kwargs()
        assert kwargs["individual_initial_knowledge"] is True
        assert kwargs["init_knowledge_strategy"] == "joint"


def _kc_data(student_ids, covariates, columns=("pretest", "age")) -> KCData:
    n = len(student_ids)
    return KCData(
        correctness=np.ones((n, 3), dtype=np.int8),
        student_inter_dict={
            s: StudentInteraction(problem_ids=["1", "2", "3"], length=3)
            for s in student_ids
        },
        lengths=np.full(n, 3, dtype=np.int32),
        student_ids=list(student_ids),
        problem_ids=["1", "2", "3"],
        groups=np.ones(n, dtype=np.int32),
        group_2_index={"g": 1},
        covariates=np.asarray(covariates, dtype=np.float64),
        covariate_columns=list(columns),
    )


def _fake_fit(n_draws=200, b0=0.5, b1=(1.0, -0.5), sigma=0.7, n_train=3):
    rng = np.random.default_rng(0)
    draws = {
        "pi_b0_know_param": np.full(n_draws, b0),
        "pi_b1_know_param": np.tile(np.asarray(b1), (n_draws, 1)),
        "pi_sigma_param": np.full(n_draws, sigma),
        "logit_pi_know_z": rng.normal(size=(n_draws, n_train)),
    }
    fit = SimpleNamespace(stan_variable=lambda name: draws[name])
    return fit, draws


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TestJointInitialKnowledge:
    def _fitted_model(self, monkeypatch, cls=StandardBKT):
        model = _joint_model(cls)
        _mock_fit(model, monkeypatch)
        train = _df()
        if cls is MultiBKT:
            train = train.assign(group_id="g")
        model.fit(train, student_covariates=_covariates())
        return model

    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_seen_reuse_fitted_z_and_unseen_use_regression_mean(
        self, monkeypatch, cls
    ):
        model = self._fitted_model(monkeypatch, cls)
        fit, draws = _fake_fit()
        covariates = np.array([[0.0, 11.0], [0.3, 11.5]])
        kc_data = _kc_data(["s2", "s_new"], covariates)
        out = model._individual_pi_know_draw_matrix(fit, kc_data, "default_kc")
        mean_logit = draws["pi_b0_know_param"][:, None] + covariates @ np.array([1.0, -0.5])
        seen = _sigmoid(mean_logit[:, 0] + 0.7 * draws["logit_pi_know_z"][:, 1])
        unseen = _sigmoid(mean_logit[:, 1])
        np.testing.assert_allclose(out[:, 0], seen)
        np.testing.assert_allclose(out[:, 1], unseen)

        point = model._extract_individual_pi_know_point_estimate(
            fit, kc_data, "default_kc", "mean"
        )
        np.testing.assert_allclose(point, [seen.mean(), unseen.mean()])

    def test_output_is_deterministic(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        fit, _ = _fake_fit()
        kc_data = _kc_data(["s1", "s_new"], [[0.1, 1.0], [0.2, 2.0]])
        a = model._individual_pi_know_draw_matrix(fit, kc_data, "default_kc")
        b = model._individual_pi_know_draw_matrix(fit, kc_data, "default_kc")
        np.testing.assert_array_equal(a, b)

    def test_students_are_matched_by_id_not_position(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        fit, _ = _fake_fit()
        cov = {"s1": [0.1, 1.0], "s3": [0.4, 3.0]}
        forward = _kc_data(["s1", "s3"], [cov["s1"], cov["s3"]])
        backward = _kc_data(["s3", "s1"], [cov["s3"], cov["s1"]])
        a = model._extract_individual_pi_know_point_estimate(
            fit, forward, "default_kc", "mean"
        )
        b = model._extract_individual_pi_know_point_estimate(
            fit, backward, "default_kc", "mean"
        )
        np.testing.assert_allclose(a, b[::-1])

    def test_unseen_draws_carry_only_parameter_uncertainty(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        fit, draws = _fake_fit(sigma=5.0)
        draws["pi_b0_know_param"] = np.zeros_like(draws["pi_b0_know_param"])
        draws["pi_b1_know_param"] = np.zeros_like(draws["pi_b1_know_param"])
        kc_data = _kc_data(["s_new"], [[0.0, 0.0]])
        out = model._individual_pi_know_draw_matrix(fit, kc_data, "default_kc")
        np.testing.assert_allclose(out, 0.5)

    def test_predict_succeeds_with_unseen_student(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        captured: dict = {}

        def _fake_extract(fit, n_students, point_estimate="mean", groups=None,
                          kc_data=None, kc_id=None):
            captured["student_ids"] = kc_data.student_ids
            captured["covariates"] = kc_data.covariates
            captured["kc_id"] = kc_id
            ones = np.full(n_students, 0.2)
            return ones, ones, ones * 0.5, ones * 0.5, ones * 0.5

        monkeypatch.setattr(model, "_extract_bkt_params_from_fit", _fake_extract)
        students = ("s1", "s2", "s_new")
        out = model.predict(
            _df(students), student_covariates=_covariates(students)
        )
        assert set(out["student_id"].astype(str)) == set(students)
        assert captured["covariates"].shape == (3, 2)
        assert captured["kc_id"] == "default_kc"

    def test_predict_raises_without_covariates(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        with pytest.raises(ValueError, match="student_covariates"):
            model.predict(_df())

    def test_predict_rejects_reordered_covariate_columns(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        swapped = _covariates()[["student_id", "age", "pretest"]]
        with pytest.raises(ValueError, match="do not match"):
            model.predict(_df(), student_covariates=swapped)

    def test_joint_keeps_unseen_students(self, monkeypatch):
        model = self._fitted_model(monkeypatch)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        assert len(model._drop_unseen_students(data, mapping)) == len(data)


def _correctness_only_model(monkeypatch, cls=StandardBKT):
    model = cls(individual_initial_knowledge=True)
    _mock_fit(model, monkeypatch)
    train = _df()
    if cls is MultiBKT:
        train = train.assign(group_id="g")
    model.fit(train)
    return model


class TestCorrectnessOnlyIndividual:
    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_seen_students_matched_by_id_in_any_order(self, monkeypatch, cls):
        model = _correctness_only_model(monkeypatch, cls)
        logits = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]])
        fit = SimpleNamespace(stan_variable=lambda name: logits)
        forward = _kc_data(["s1", "s3"], [[0.0], [0.0]], columns=())
        backward = _kc_data(["s3", "s1"], [[0.0], [0.0]], columns=())
        a = model._individual_pi_know_draw_matrix(fit, forward, "default_kc")
        b = model._individual_pi_know_draw_matrix(fit, backward, "default_kc")
        np.testing.assert_allclose(a[:, 0], _sigmoid(logits[:, 0]))
        np.testing.assert_allclose(a[:, 1], _sigmoid(logits[:, 2]))
        np.testing.assert_allclose(a, b[:, ::-1])

    def test_unseen_students_dropped_with_warning(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        data = _df(("s1", "s2", "s_new")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        with pytest.warns(UserWarning, match="1 student"):
            out = model._drop_unseen_students(data, mapping)
        assert set(out["student_id"]) == {"s1", "s2"}

    def test_long_dropped_list_is_truncated(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        new = tuple(f"n{i}" for i in range(15))
        data = _df(("s1",) + new).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        with pytest.warns(UserWarning, match="and 5 more"):
            model._drop_unseen_students(data, mapping)

    def test_all_unseen_raises(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        data = _df(("x", "y")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        with pytest.raises(ValueError, match="None of the students"):
            model._drop_unseen_students(data, mapping)

    def test_all_seen_passes_through_silently(self, monkeypatch, recwarn):
        model = _correctness_only_model(monkeypatch)
        data = _df(("s2", "s1")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        assert len(model._drop_unseen_students(data, mapping)) == len(data)
        assert not recwarn.list

    def test_posterior_inputs_drop_unseen(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        with pytest.warns(UserWarning, match="not part of the fit"):
            out, _ = _prepare_posterior_prediction_inputs(model, data)
        assert set(out["student_id"]) == {"s1"}

    def test_predict_passes_only_fitted_students(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        seen: dict = {}

        def _fake_extract(fit, n_students, point_estimate="mean", groups=None,
                          kc_data=None, kc_id=None):
            seen["student_ids"] = kc_data.student_ids
            ones = np.full(n_students, 0.2)
            return ones, ones, ones * 0.5, ones * 0.5, ones * 0.5

        monkeypatch.setattr(model, "_extract_bkt_params_from_fit", _fake_extract)
        with pytest.warns(UserWarning, match="not part of the fit"):
            out = model.predict(_df(("s1", "s2", "s_new")))
        assert seen["student_ids"] == ["s1", "s2"]
        assert set(out["student_id"].astype(str)) == {"s1", "s2"}

    def test_predict_all_unseen_raises(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        with pytest.raises(ValueError, match="None of the students"):
            model.predict(_df(("x", "y")))


class TestDefaultConfigurationUnchanged:
    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_unseen_students_not_filtered(self, monkeypatch, recwarn, cls):
        model = cls()
        _mock_fit(model, monkeypatch)
        train = _df()
        if cls is MultiBKT:
            train = train.assign(group_id="g")
        model.fit(train)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        out = model._drop_unseen_students(data, mapping)
        assert out is data
        assert not recwarn.list

    def test_all_unseen_does_not_raise(self, monkeypatch, recwarn):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        data = _df(("x", "y")).assign(kc_id="default_kc")
        mapping = ColumnNames.apply_default_mapping(None)
        assert len(model._drop_unseen_students(data, mapping)) == len(data)
        assert not recwarn.list

    def test_posterior_inputs_keep_unseen_students(self, monkeypatch, recwarn):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        out, _ = _prepare_posterior_prediction_inputs(model, data)
        assert set(out["student_id"]) == {"s1", "s_new"}
        assert not recwarn.list


class TestGQDataForMixedStudents:
    def test_gq_data_dict_contents(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), student_covariates=_covariates())

        gq_model = MagicMock()
        gq_model.generate_quantities.return_value = MagicMock()
        fit = MagicMock()
        fit_z = np.zeros((10, 3))
        monkeypatch.setattr(model.fits, "get_fit", lambda kc: fit)
        monkeypatch.setattr(
            model,
            "_extract_named_param_matrix",
            lambda f, name: fit_z,
        )

        students = ("s2", "s_new", "s1", "s_new2")
        predict_cov = _covariates(students)
        model._predict_generated_quantities(
            data=_df(students),
            gq_model=gq_model,
            column_mapping=None,
            student_covariates=predict_cov,
        )
        data = gq_model.generate_quantities.call_args.kwargs["data"]

        assert data["nStudents"] == 4
        assert data["nTrainStudents"] == 3
        assert data["covariates"].shape == (4, 2)
        assert "covariates_new" not in data
        # iter_kc_data sorts students naturally: s1, s2, s_new, s_new2
        np.testing.assert_array_equal(data["train_student_idx"], [1, 2, 0, 0])
        expected = predict_cov.set_index("student_id").loc[
            ["s1", "s2", "s_new", "s_new2"], ["pretest", "age"]
        ]
        np.testing.assert_allclose(data["covariates"], expected.to_numpy())

    def test_correctness_only_gq_maps_students_by_id(self, monkeypatch):
        model = StandardBKT(individual_initial_knowledge=True)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        gq_model = MagicMock()
        monkeypatch.setattr(model.fits, "get_fit", lambda kc: MagicMock())
        monkeypatch.setattr(
            model, "_extract_named_param_matrix", lambda f, name: np.zeros((10, 3))
        )
        model._predict_generated_quantities(
            data=_df(("s3", "s1")), gq_model=gq_model, column_mapping=None
        )
        data = gq_model.generate_quantities.call_args.kwargs["data"]
        assert data["nTrainStudents"] == 3
        np.testing.assert_array_equal(data["train_student_idx"], [1, 3])

    def test_non_joint_gq_data_untouched(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        gq_model = MagicMock()
        monkeypatch.setattr(model.fits, "get_fit", lambda kc: MagicMock())
        model._predict_generated_quantities(
            data=_df(), gq_model=gq_model, column_mapping=None
        )
        data = gq_model.generate_quantities.call_args.kwargs["data"]
        assert data["joint_pi_know"] == 0
        assert data["nTrainStudents"] == data["nStudents"]
        np.testing.assert_array_equal(
            data["train_student_idx"], np.arange(1, data["nStudents"] + 1)
        )


def _kc_df(kc_students: dict) -> pd.DataFrame:
    return pd.concat(
        [_df(students).assign(kc_id=kc) for kc, students in kc_students.items()],
        ignore_index=True,
    )


def _fit_students(model, monkeypatch, kc_students, covariates=None):
    _mock_fit(model, monkeypatch)
    model.fit(_kc_df(kc_students), student_covariates=covariates)
    return model


def _no_fit_warning(recwarn) -> bool:
    return not [w for w in recwarn.list if "not part of the fit" in str(w.message)]


class TestMultiKCStudents:
    fit_students = {"kc_a": ("s1", "s2"), "kc_b": ("s2", "s3")}

    def test_student2index_is_per_kc(self, monkeypatch):
        model = _fit_students(
            _joint_model(), monkeypatch, self.fit_students, _covariates()
        )
        assert model.fits.get_fit_save_entry("kc_a").student2index == {"s1": 1, "s2": 2}
        assert model.fits.get_fit_save_entry("kc_b").student2index == {"s2": 1, "s3": 2}

    def test_correctness_only_drop_is_per_kc(self, monkeypatch):
        model = _fit_students(
            StandardBKT(individual_initial_knowledge=True),
            monkeypatch,
            self.fit_students,
        )
        data = _kc_df({"kc_a": ("s1", "s3"), "kc_b": ("s1", "s3")})
        mapping = ColumnNames.apply_default_mapping(None)
        with pytest.warns(UserWarning, match="2 student"):
            out = model._drop_unseen_students(data, mapping)
        assert set(zip(out["kc_id"], out["student_id"])) == {
            ("kc_a", "s1"),
            ("kc_b", "s3"),
        }

    def test_kc_with_only_unseen_students_is_dropped_others_remain(self, monkeypatch):
        model = _fit_students(
            StandardBKT(individual_initial_knowledge=True),
            monkeypatch,
            self.fit_students,
        )
        data = _kc_df({"kc_a": ("s3",), "kc_b": ("s3",)})
        mapping = ColumnNames.apply_default_mapping(None)
        with pytest.warns(UserWarning, match="1 student"):
            out = model._drop_unseen_students(data, mapping)
        assert set(out["kc_id"]) == {"kc_b"}

    def test_joint_keeps_unseen_students_in_every_kc(self, monkeypatch, recwarn):
        model = _fit_students(
            _joint_model(), monkeypatch, self.fit_students, _covariates()
        )
        data = _kc_df({"kc_a": ("s1", "s3"), "kc_b": ("s1", "s3")})
        mapping = ColumnNames.apply_default_mapping(None)
        assert len(model._drop_unseen_students(data, mapping)) == len(data)
        assert _no_fit_warning(recwarn)

    def test_joint_draw_matrix_uses_each_kcs_own_index(self, monkeypatch):
        model = _fit_students(
            _joint_model(), monkeypatch, self.fit_students, _covariates()
        )
        fit_a, draws_a = _fake_fit(n_train=2)
        fit_b, draws_b = _fake_fit(n_train=2)
        draws_b["logit_pi_know_z"] = -draws_b["logit_pi_know_z"] + 0.3
        cov = np.array([[0.1, 1.0], [0.4, 3.0]])
        kc_data = _kc_data(["s1", "s3"], cov)
        b1 = np.array([1.0, -0.5])
        mean_logit = 0.5 + cov @ b1

        out_a = model._individual_pi_know_draw_matrix(fit_a, kc_data, "kc_a")
        out_b = model._individual_pi_know_draw_matrix(fit_b, kc_data, "kc_b")
        # s1 is seen only in kc_a, s3 is seen only in kc_b
        np.testing.assert_allclose(
            out_a[:, 0], _sigmoid(mean_logit[0] + 0.7 * draws_a["logit_pi_know_z"][:, 0])
        )
        np.testing.assert_allclose(out_a[:, 1], _sigmoid(mean_logit[1]))
        np.testing.assert_allclose(out_b[:, 0], _sigmoid(mean_logit[0]))
        np.testing.assert_allclose(
            out_b[:, 1], _sigmoid(mean_logit[1] + 0.7 * draws_b["logit_pi_know_z"][:, 1])
        )

    def test_gq_data_dict_maps_students_per_kc(self, monkeypatch):
        model = _fit_students(
            _joint_model(), monkeypatch, self.fit_students, _covariates()
        )
        fits = {
            "kc_a": SimpleNamespace(z=np.zeros((10, 2))),
            "kc_b": SimpleNamespace(z=np.zeros((10, 2))),
        }
        monkeypatch.setattr(model.fits, "get_fit", lambda kc: fits[kc])
        monkeypatch.setattr(model, "_extract_named_param_matrix", lambda f, name: f.z)
        gq_model = MagicMock()
        students = ("s1", "s3")
        model._predict_generated_quantities(
            data=_kc_df({"kc_a": students, "kc_b": students}),
            gq_model=gq_model,
            column_mapping=None,
            student_covariates=_covariates(students),
        )
        calls = [c.kwargs["data"] for c in gq_model.generate_quantities.call_args_list]
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0]["train_student_idx"], [1, 0])
        np.testing.assert_array_equal(calls[1]["train_student_idx"], [0, 2])


class TestCustomStudentColumn:
    mapping = {"student_id": "sid"}

    def _renamed(self, df):
        return df.rename(columns={"student_id": "sid"})

    def test_joint_fit_follows_column_mapping(self, monkeypatch):
        model = _joint_model()
        captured = _mock_fit(model, monkeypatch)
        model.fit(
            self._renamed(_df()),
            column_mapping=self.mapping,
            student_covariates=self._renamed(_covariates()),
        )
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"s1": 1, "s2": 2, "s3": 3}
        assert entry.covariate_columns == ("pretest", "age")
        assert captured[0]["covariates"].shape == (3, 2)

    def test_covariates_must_use_the_mapped_id_column(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="missing the student ID column 'sid'"):
            model.fit(
                self._renamed(_df()),
                column_mapping=self.mapping,
                student_covariates=_covariates(),
            )

    def test_correctness_only_drop_follows_column_mapping(self, monkeypatch):
        model = StandardBKT(individual_initial_knowledge=True)
        _mock_fit(model, monkeypatch)
        model.fit(self._renamed(_df()), column_mapping=self.mapping)
        mapping = ColumnNames.apply_default_mapping(self.mapping)
        data = self._renamed(_df(("s1", "s_new"))).assign(kc_id="default_kc")
        with pytest.warns(UserWarning, match="1 student"):
            out = model._drop_unseen_students(data, mapping)
        assert set(out["sid"]) == {"s1"}


class TestIntegerAndStringIds:
    def _int_df(self):
        return _df(("1", "2", "3")).assign(student_id=lambda d: d["student_id"].astype(int))

    def _int_covariates(self, as_str=False):
        ids = ["1", "2", "3"] if as_str else [1, 2, 3]
        return pd.DataFrame(
            {"student_id": ids, "pretest": [0.1, 0.2, 0.3], "age": [10.0, 11.0, 12.0]}
        )

    @pytest.mark.parametrize("as_str", [False, True])
    def test_joint_fit_matches_ids_across_dtypes(self, monkeypatch, as_str):
        model = _joint_model()
        captured = _mock_fit(model, monkeypatch)
        model.fit(self._int_df(), student_covariates=self._int_covariates(as_str))
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"1": 1, "2": 2, "3": 3}
        np.testing.assert_allclose(captured[0]["covariates"][:, 0], [0.1, 0.2, 0.3])

    def test_draw_matrix_matches_int_ids_by_string(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        cov = self._int_covariates()
        model.fit(self._int_df(), student_covariates=cov)
        fit, draws = _fake_fit()
        indexed = cov.astype({"student_id": str}).set_index("student_id")
        kc_data = format_kc_data(
            self._int_df().query("student_id == 2"),
            student_covariates=indexed,
            covariate_columns=["pretest", "age"],
        )["default_kc"]
        assert kc_data.student_ids == ["2"]
        out = model._individual_pi_know_draw_matrix(fit, kc_data, "default_kc")
        mean_logit = 0.5 + np.array([0.2, 11.0]) @ np.array([1.0, -0.5])
        np.testing.assert_allclose(
            out[:, 0], _sigmoid(mean_logit + 0.7 * draws["logit_pi_know_z"][:, 1])
        )

    def test_correctness_only_drop_matches_int_and_str_ids(self, monkeypatch, recwarn):
        model = StandardBKT(individual_initial_knowledge=True)
        _mock_fit(model, monkeypatch)
        model.fit(self._int_df())
        mapping = ColumnNames.apply_default_mapping(None)
        as_int = self._int_df().assign(kc_id="default_kc")
        as_str = as_int.assign(student_id=lambda d: d["student_id"].astype(str))
        assert len(model._drop_unseen_students(as_int, mapping)) == len(as_int)
        assert len(model._drop_unseen_students(as_str, mapping)) == len(as_str)
        assert _no_fit_warning(recwarn)
        unseen = as_int.assign(student_id=lambda d: d["student_id"] + 10)
        with pytest.raises(ValueError, match="None of the students"):
            model._drop_unseen_students(unseen, mapping)


class TestOverwriteRefit:
    def test_refit_replaces_student_index_and_covariate_columns(self, monkeypatch):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), student_covariates=_covariates())
        new_students = ("s4", "s5")
        new_cov = _covariates(new_students)[["student_id", "pretest"]]

        with pytest.raises(ValueError, match="already exists"):
            model.fit(_df(new_students), student_covariates=new_cov)
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"s1": 1, "s2": 2, "s3": 3}

        model.fit(_df(new_students), student_covariates=new_cov, overwrite_kcs=True)
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"s4": 1, "s5": 2}
        assert entry.covariate_columns == ("pretest",)
        assert model.fits.num_fitted_kcs == 1
        model._check_covariate_columns_match("default_kc", ["pretest"])
        with pytest.raises(ValueError, match="do not match"):
            model._check_covariate_columns_match("default_kc", ["pretest", "age"])


class _DummySavedFit:
    def save_csvfiles(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8") as f:
            f.write("lp__\n0\n")


class TestLowMemoryJoint:
    def _evicted_model(self, monkeypatch):
        model = _joint_model(low_memory=True)
        _mock_fit(model, monkeypatch)
        monkeypatch.setattr(
            model, "_fit_stan_model_using_method", lambda data_dict, fit_options: _DummySavedFit()
        )
        model.fit(_df(), student_covariates=_covariates())
        return model

    def test_eviction_keeps_student_index_and_covariate_columns(self, monkeypatch):
        model = self._evicted_model(monkeypatch)
        assert "default_kc" not in model.fits.stan_fits
        entry = model.fits.get_fit_save_entry("default_kc")
        assert entry.student2index == {"s1": 1, "s2": 2, "s3": 3}
        assert entry.covariate_columns == ("pretest", "age")

    def test_predict_after_lazy_reload_matches_students_by_id(self, monkeypatch):
        model = self._evicted_model(monkeypatch)
        fit, draws = _fake_fit()
        draws.update(
            {name: np.full(200, value) for name, value in
             [("learn", 0.3), ("forget", 0.1), ("guess", 0.2), ("slip", 0.1)]}
        )
        monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", lambda _: fit)

        students = ("s3", "s_new", "s1")
        covariates = _covariates(students)
        out = model.predict(
            _df(students),
            student_covariates=covariates,
            parallel=False,
            fast_math=False,
        )
        assert "default_kc" in model.fits.stan_fits

        first = out.groupby(out["student_id"].astype(str), observed=True)["pKnow"].first()
        b1 = np.array([1.0, -0.5])
        x = covariates.set_index("student_id")[["pretest", "age"]]
        mean_logit = {s: 0.5 + x.loc[s].to_numpy() @ b1 for s in students}
        z = draws["logit_pi_know_z"]
        expected = {
            "s1": _sigmoid(mean_logit["s1"] + 0.7 * z[:, 0]).mean(),
            "s3": _sigmoid(mean_logit["s3"] + 0.7 * z[:, 2]).mean(),
            "s_new": _sigmoid(mean_logit["s_new"]),
        }
        for student, value in expected.items():
            assert first.loc[student] == pytest.approx(value)


def _patch_posterior_backends(model, monkeypatch) -> list:
    seen: list = []

    def _record(data, *args, **kwargs):
        seen.append(data)
        return {}

    monkeypatch.setattr(
        predictions_module, "_get_or_compile_gq_model", lambda m, smoothed: MagicMock()
    )
    monkeypatch.setattr(
        model, "_predict_generated_quantities", lambda data, **kw: _record(data)
    )
    monkeypatch.setattr(model, "_process_predict_gq", lambda raw, data, mapping: {})
    monkeypatch.setattr(
        model, "_predict_summary_streaming", lambda data, **kw: _record(data)
    )
    monkeypatch.setattr(
        predictions_module,
        "_predict_posterior_draws_numba",
        lambda model, data, **kw: _record(data),
    )
    return seen


SMOOTHED_POSTERIOR_CALLS = [
    ("predict_smoothed_posterior_stan", {}),
    ("predict_smoothed_posterior_draws", {}),
    ("predict_smoothed_posterior_draws", {"backend": "numba"}),
    ("predict_smoothed_posterior_summary", {}),
]


def _stub_extract(model, monkeypatch, seen):
    def _fake_extract(fit, n_students, point_estimate="mean", groups=None,
                      kc_data=None, kc_id=None):
        seen["student_ids"] = kc_data.student_ids
        ones = np.full(n_students, 0.2)
        return ones, ones, ones * 0.5, ones * 0.5, ones * 0.5

    monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
    monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
    monkeypatch.setattr(model, "_extract_bkt_params_from_fit", _fake_extract)


class TestSmoothedPredictionRules:
    def test_predict_smoothed_drops_unseen_for_correctness_only(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        seen: dict = {}
        _stub_extract(model, monkeypatch, seen)
        with pytest.warns(UserWarning, match="not part of the fit"):
            out = model.predict_smoothed(_df(("s1", "s2", "s_new")))
        assert seen["student_ids"] == ["s1", "s2"]
        assert set(out["student_id"].astype(str)) == {"s1", "s2"}

    def test_predict_smoothed_all_unseen_raises(self, monkeypatch):
        model = _correctness_only_model(monkeypatch)
        _stub_extract(model, monkeypatch, {})
        with pytest.raises(ValueError, match="None of the students"):
            model.predict_smoothed(_df(("x", "y")))

    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_predict_smoothed_default_config_unchanged(self, monkeypatch, recwarn, cls):
        model = cls()
        _mock_fit(model, monkeypatch)
        train = _df() if cls is StandardBKT else _df().assign(group_id="g")
        model.fit(train)
        seen: dict = {}
        _stub_extract(model, monkeypatch, seen)
        students = ("x", "y") if cls is StandardBKT else ("s1", "s_new")
        data = _df(students) if cls is StandardBKT else _df(students).assign(group_id="g")
        out = model.predict_smoothed(data)
        assert set(out["student_id"].astype(str)) == set(students)
        assert _no_fit_warning(recwarn)

    def test_predict_smoothed_joint_keeps_unseen(self, monkeypatch, recwarn):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), student_covariates=_covariates())
        seen: dict = {}
        _stub_extract(model, monkeypatch, seen)
        students = ("s1", "s_new")
        out = model.predict_smoothed(
            _df(students), student_covariates=_covariates(students)
        )
        assert set(out["student_id"].astype(str)) == set(students)
        assert _no_fit_warning(recwarn)

    @pytest.mark.parametrize("method,kwargs", SMOOTHED_POSTERIOR_CALLS)
    def test_smoothed_posterior_drops_unseen_for_correctness_only(
        self, monkeypatch, method, kwargs
    ):
        model = _correctness_only_model(monkeypatch)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        seen = _patch_posterior_backends(model, monkeypatch)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        with pytest.warns(UserWarning, match="not part of the fit"):
            getattr(model, method)(data, **kwargs)
        assert set(seen[0]["student_id"]) == {"s1"}

    @pytest.mark.parametrize("method,kwargs", SMOOTHED_POSTERIOR_CALLS)
    def test_smoothed_posterior_all_unseen_raises(self, monkeypatch, method, kwargs):
        model = _correctness_only_model(monkeypatch)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        _patch_posterior_backends(model, monkeypatch)
        data = _df(("x", "y")).assign(kc_id="default_kc")
        with pytest.raises(ValueError, match="None of the students"):
            getattr(model, method)(data, **kwargs)

    @pytest.mark.parametrize("method,kwargs", SMOOTHED_POSTERIOR_CALLS)
    def test_smoothed_posterior_default_config_unchanged(
        self, monkeypatch, recwarn, method, kwargs
    ):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        seen = _patch_posterior_backends(model, monkeypatch)
        data = _df(("s1", "s_new")).assign(kc_id="default_kc")
        getattr(model, method)(data, **kwargs)
        assert set(seen[0]["student_id"]) == {"s1", "s_new"}
        assert _no_fit_warning(recwarn)

    @pytest.mark.parametrize("method,kwargs", SMOOTHED_POSTERIOR_CALLS)
    def test_smoothed_posterior_joint_keeps_unseen(self, monkeypatch, recwarn, method, kwargs):
        model = _joint_model()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), student_covariates=_covariates())
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        seen = _patch_posterior_backends(model, monkeypatch)
        students = ("s1", "s_new")
        data = _df(students).assign(kc_id="default_kc")
        getattr(model, method)(
            data, student_covariates=_covariates(students), **kwargs
        )
        assert set(seen[0]["student_id"]) == set(students)
        assert _no_fit_warning(recwarn)
