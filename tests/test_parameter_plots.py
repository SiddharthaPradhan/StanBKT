import numpy as np
import pandas as pd
import pytest

import stanbkt.plot.parameter_plots as parameter_plots


def _fake_idata():
    class _Posterior:
        def __init__(self) -> None:
            self._data = {
                "pi_know": np.random.normal(size=(2, 20)),
                "learn": np.random.normal(size=(2, 20)),
                "guess": np.random.normal(size=(2, 20)),
            }
            self.data_vars = self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    class _IData:
        def __init__(self) -> None:
            self.posterior = _Posterior()

    return _IData()


def _fake_indexed_idata():
    class _Posterior:
        def __init__(self) -> None:
            self._data = {
                "pi_know[1]": np.random.normal(size=(2, 20)),
                "pi_know[2]": np.random.normal(size=(2, 20)),
                "learn[1]": np.random.normal(size=(2, 20)),
                "learn[2]": np.random.normal(size=(2, 20)),
            }
            self.data_vars = self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    class _IData:
        def __init__(self) -> None:
            self.posterior = _Posterior()

    return _IData()


def test_plot_dist_uses_dist_only(monkeypatch) -> None:
    idata = _fake_idata()
    monkeypatch.setattr(parameter_plots.az, "from_cmdstanpy", lambda posterior: idata)

    captured: dict[str, object] = {}

    plot_object = object()

    def _fake_dist(idata_arg, var_names=None, **kwargs):
        captured["idata"] = idata_arg
        captured["var_names"] = var_names
        captured["kwargs"] = kwargs
        return plot_object

    monkeypatch.setattr(parameter_plots.az, "plot_dist", _fake_dist)

    result = parameter_plots.plot_dist(
        fit=object(), params=["pi_know", "learn"], ci_prob=0.8, ci_kind="eti"
    )

    assert result is plot_object
    assert captured["idata"] is idata
    assert captured["var_names"] == ["pi_know", "learn"]
    assert captured["kwargs"]["ci_prob"] == 0.8
    assert captured["kwargs"]["ci_kind"] == "eti"
    assert captured["kwargs"]["backend"] == "matplotlib"
    assert captured["kwargs"]["col_wrap"] == 3
    assert "figure_kwargs" not in captured["kwargs"]
    assert "visuals" not in captured["kwargs"]


def test_plot_dist_raises_on_unknown_param(monkeypatch) -> None:
    idata = _fake_idata()
    monkeypatch.setattr(parameter_plots.az, "from_cmdstanpy", lambda posterior: idata)

    with pytest.raises(ValueError, match=r"Unknown parameter\(s\)"):
        parameter_plots.plot_dist(fit=object(), params=["not_a_param"])


def test_plot_trace_passes_var_names(monkeypatch) -> None:
    idata = _fake_idata()
    monkeypatch.setattr(parameter_plots.az, "from_cmdstanpy", lambda posterior: idata)

    captured: dict[str, object] = {}
    plot_object = object()

    def _fake_trace(idata_arg, var_names=None, **kwargs):
        captured["idata"] = idata_arg
        captured["var_names"] = var_names
        captured["kwargs"] = kwargs
        return plot_object

    monkeypatch.setattr(parameter_plots.az, "plot_trace", _fake_trace)

    result = parameter_plots.plot_trace(
        fit=object(), params=["learn", "guess"], col_wrap=2
    )

    assert result is plot_object
    assert captured["idata"] is idata
    assert captured["var_names"] == ["learn", "guess"]
    assert captured["kwargs"]["backend"] == "matplotlib"
    assert captured["kwargs"]["col_wrap"] == 2
    assert "figure_kwargs" not in captured["kwargs"]
    assert "visuals" not in captured["kwargs"]


def test_plot_trace_supports_vb_fit(monkeypatch) -> None:
    class _DummyVB:
        column_names = ["lp__", "pi_know[1]", "learn[1]", "stepsize__"]

        def __init__(self) -> None:
            self.variational_sample_pd = pd.DataFrame(
                {
                    "lp__": [-1.0, -0.9, -1.1],
                    "pi_know[1]": [0.1, 0.2, 0.3],
                    "learn[1]": [0.05, 0.06, 0.07],
                    "stepsize__": [1.0, 1.0, 1.0],
                }
            )

    idata = _fake_idata()
    monkeypatch.setattr(parameter_plots, "CmdStanVB", _DummyVB)
    monkeypatch.setattr(
        parameter_plots.az,
        "from_cmdstanpy",
        lambda posterior: (_ for _ in ()).throw(
            AssertionError("unexpected from_cmdstanpy call")
        ),
    )

    captured: dict[str, object] = {}

    def _fake_from_dict(data):
        captured["posterior"] = data["posterior"]
        return idata

    monkeypatch.setattr(parameter_plots.az, "from_dict", _fake_from_dict)
    monkeypatch.setattr(
        parameter_plots.az,
        "plot_trace",
        lambda idata_arg, var_names=None, **kwargs: {
            "idata": idata_arg,
            "var_names": var_names,
            "kwargs": kwargs,
        },
    )

    result = parameter_plots.plot_trace(fit=_DummyVB(), params=["pi_know", "learn"])

    assert result["idata"] is idata
    assert result["var_names"] == ["pi_know", "learn"]
    assert sorted(captured["posterior"].keys()) == ["learn", "lp__", "pi_know"]
    assert captured["posterior"]["pi_know"].shape == (1, 3)


def test_plot_dist_rejects_mle_fit(monkeypatch) -> None:
    class _DummyMLE:
        pass

    monkeypatch.setattr(parameter_plots, "CmdStanMLE", _DummyMLE)

    with pytest.raises(ValueError, match="do not contain posterior draws"):
        parameter_plots.plot_dist(fit=_DummyMLE())


def test_plot_trace_expands_base_names_for_indexed_params(monkeypatch) -> None:
    idata = _fake_indexed_idata()
    monkeypatch.setattr(parameter_plots.az, "from_cmdstanpy", lambda posterior: idata)

    captured: dict[str, object] = {}

    def _fake_trace(idata_arg, var_names=None, **kwargs):
        captured["idata"] = idata_arg
        captured["var_names"] = var_names
        return object()

    monkeypatch.setattr(parameter_plots.az, "plot_trace", _fake_trace)

    parameter_plots.plot_trace(fit=object(), params=["pi_know"])

    assert captured["idata"] is idata
    assert captured["var_names"] == ["pi_know[1]", "pi_know[2]"]
