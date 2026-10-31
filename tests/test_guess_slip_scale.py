"""Tests for the configurable guess and slip upper bound (mock based, no CmdStan)."""

from __future__ import annotations

import numpy as np
import pytest

from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.data_utils import KCData, StudentInteraction
from stanbkt.utils.model_io import _parse_model_init_kwargs
from stanbkt.utils.verbose import VerbosityLevel


def _kc_data(n: int = 3) -> KCData:
    return KCData(
        correctness=np.ones((n, 3), dtype=np.int8),
        student_inter_dict={
            str(s): StudentInteraction(problem_ids=["1", "2", "3"], length=3)
            for s in range(n)
        },
        lengths=np.full(n, 3, dtype=np.int32),
        student_ids=[str(s) for s in range(n)],
        problem_ids=["1", "2", "3"],
        groups=np.ones(n, dtype=np.int32),
        group_2_index={"g": 1},
    )


@pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
def test_default_is_half(cls):
    model = cls()
    assert model.guess_slip_scale == 0.5
    assert model._build_stan_data_dict(_kc_data())["guess_slip_scale"] == 0.5


@pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
def test_custom_scale_reaches_stan_data_and_init_kwargs(cls):
    model = cls(guess_slip_scale=0.3)
    assert model._build_stan_data_dict(_kc_data())["guess_slip_scale"] == 0.3
    assert model._get_model_init_kwargs()["guess_slip_scale"] == 0.3


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
def test_invalid_scale_raises(scale):
    with pytest.raises(ValueError, match="guess_slip_scale"):
        StandardBKT(guess_slip_scale=scale)


def test_unit_scale_is_allowed():
    assert StandardBKT(guess_slip_scale=1.0).guess_slip_scale == 1.0


def _raw_init_kwargs(**extra):
    return {
        "fit_method": "mcmc",
        "verbose": int(VerbosityLevel.INFO),
        "stan_compile_kwargs": {},
        "cpp_compile_kwargs": {},
        **extra,
    }


def test_load_defaults_scale_for_older_archives():
    assert _parse_model_init_kwargs(_raw_init_kwargs())["guess_slip_scale"] == 0.5


def test_load_restores_saved_scale():
    parsed = _parse_model_init_kwargs(_raw_init_kwargs(guess_slip_scale=0.25))
    assert parsed["guess_slip_scale"] == 0.25


def test_load_rejects_non_numeric_scale():
    with pytest.raises(ValueError, match="guess_slip_scale"):
        _parse_model_init_kwargs(_raw_init_kwargs(guess_slip_scale="0.5"))
