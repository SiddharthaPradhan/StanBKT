import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_options import MCMCFitOptions, MLEFitOptions
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.model_io import load_model
from stanbkt.utils.verbose import (
    VerboseMixin,
    VerbosityLevel,
    quiet_when_silent,
    silence_cmdstanpy,
)


class TestVerbosityLevel:
    def test_info_value(self):
        assert VerbosityLevel.INFO.value == 2

    def test_warn_value(self):
        assert VerbosityLevel.WARN.value == 1

    def test_debug_value(self):
        assert VerbosityLevel.DEBUG.value == 3

    def test_all_members_present(self):
        names = {m.name for m in VerbosityLevel}
        assert names == {"SILENT", "INFO", "WARN", "DEBUG"}

    def test_ordering(self):
        assert (
            VerbosityLevel.WARN.value
            < VerbosityLevel.INFO.value
            < VerbosityLevel.DEBUG.value
        )


class TestVerboseMixinInit:
    def test_default_verbose_is_info(self):
        m = VerboseMixin()
        assert m.verbose == VerbosityLevel.INFO

    def test_explicit_verbose_info(self):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        assert m.verbose == VerbosityLevel.INFO

    def test_explicit_verbose_warn(self):
        m = VerboseMixin(verbose=VerbosityLevel.WARN)
        assert m.verbose == VerbosityLevel.WARN

    def test_explicit_verbose_debug(self):
        m = VerboseMixin(verbose=VerbosityLevel.DEBUG)
        assert m.verbose == VerbosityLevel.DEBUG

    def test_mro_kwargs_forwarded(self):
        """VerboseMixin should forward unexpected kwargs up the MRO cleanly."""

        class Base:
            def __init__(self, **kwargs):
                self.extra = kwargs.get("extra")

        class Mixed(VerboseMixin, Base):
            pass

        obj = Mixed(verbose=VerbosityLevel.WARN, extra="hello")
        assert obj.verbose == VerbosityLevel.WARN
        assert obj.extra == "hello"


class TestVerboseMixinPrint:
    def test_info_message_printed_at_info_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("hello info")
        assert "hello info" in capsys.readouterr().out

    def test_info_message_printed_at_debug_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.DEBUG)
        m.log("hello info", level=VerbosityLevel.INFO)
        assert "hello info" in capsys.readouterr().out

    def test_debug_message_suppressed_at_info_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("secret debug", level=VerbosityLevel.DEBUG)
        assert capsys.readouterr().out == ""

    def test_debug_message_suppressed_at_warn_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.WARN)
        m.log("secret debug", level=VerbosityLevel.DEBUG)
        assert capsys.readouterr().out == ""

    def test_warn_message_printed_at_info_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("a warning", level=VerbosityLevel.WARN)
        out = capsys.readouterr().out
        assert out.startswith("WARNING: ")
        assert "a warning" in out

    def test_warn_message_printed_at_warn_level(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.WARN)
        m.log("a warning", level=VerbosityLevel.WARN)
        assert "a warning" in capsys.readouterr().out

    def test_warn_prefix_added(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.WARN)
        m.log("something wrong", level=VerbosityLevel.WARN)
        out = capsys.readouterr().out
        assert out.startswith("WARNING: ")
        assert "something wrong" in out

    def test_debug_prefix_added(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.DEBUG)
        m.log("debug msg", level=VerbosityLevel.DEBUG)
        out = capsys.readouterr().out
        assert out.startswith("DEBUG: ")
        assert "debug msg" in out


class TestSetVerbosity:
    def test_set_verbosity_to_info(self):
        m = VerboseMixin()
        m.set_verbosity(VerbosityLevel.INFO)
        assert m.verbose == VerbosityLevel.INFO

    def test_set_verbosity_to_warn(self):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.set_verbosity(VerbosityLevel.WARN)
        assert m.verbose == VerbosityLevel.WARN

    def test_set_verbosity_to_debug(self):
        m = VerboseMixin()
        m.set_verbosity(VerbosityLevel.DEBUG)
        assert m.verbose == VerbosityLevel.DEBUG

    def test_set_verbosity_raises_on_negative_level(self):
        m = VerboseMixin()
        with pytest.raises(ValueError, match="Invalid verbosity level"):
            m.set_verbosity(-1)

    def test_set_verbosity_raises_on_invalid_string(self):
        m = VerboseMixin()
        with pytest.raises(ValueError, match="Invalid verbosity level"):
            m.set_verbosity("invalid")  # type: ignore

    def test_set_verbosity_changes_print_behavior(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("test", level=VerbosityLevel.DEBUG)
        out = capsys.readouterr().out
        assert out == ""  # Should not print DEBUG at INFO level

        m.set_verbosity(VerbosityLevel.DEBUG)
        m.log("test", level=VerbosityLevel.DEBUG)
        out = capsys.readouterr().out
        assert "test" in out  # Should print DEBUG at DEBUG level

    def test_info_message_has_no_prefix(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("plain info", level=VerbosityLevel.INFO)
        out = capsys.readouterr().out
        assert not out.startswith("WARNING:")
        assert not out.startswith("DEBUG:")
        assert "plain info" in out

    def test_debug_verbose_receives_all_levels(self, capsys):
        m = VerboseMixin(verbose=VerbosityLevel.DEBUG)
        m.log("info msg", level=VerbosityLevel.INFO)
        m.log("warn msg", level=VerbosityLevel.WARN)
        m.log("debug msg", level=VerbosityLevel.DEBUG)
        out = capsys.readouterr().out
        assert "info msg" in out
        assert "warn msg" in out
        assert "debug msg" in out

    def test_default_level_is_info(self, capsys):
        """Calling _print without a level argument should use INFO."""
        m = VerboseMixin(verbose=VerbosityLevel.INFO)
        m.log("no level arg")
        assert "no level arg" in capsys.readouterr().out


class TestLevelConversion:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, VerbosityLevel.SILENT),
            (1, VerbosityLevel.WARN),
            (3, VerbosityLevel.DEBUG),
            (4, VerbosityLevel.DEBUG),
            (999, VerbosityLevel.DEBUG),
            (np.int64(7), VerbosityLevel.DEBUG),
        ],
    )
    def test_integers(self, value, expected):
        assert VerboseMixin(verbose=value).verbose is expected
        m = VerboseMixin()
        m.set_verbosity(value)
        assert m.verbose is expected

    @pytest.mark.parametrize("value", [-1, 4.5, "debug", None])
    def test_invalid_values_raise(self, value):
        with pytest.raises(ValueError, match="Invalid verbosity level"):
            VerboseMixin(verbose=value)


class TestSilent:
    @pytest.mark.parametrize("level", list(VerbosityLevel))
    def test_nothing_is_printed(self, capsys, level):
        VerboseMixin(verbose=0).log("hidden", level=level)
        assert capsys.readouterr().out == ""

    def test_logger_disabled_inside_and_restored_after(self):
        logger = logging.getLogger("cmdstanpy")
        assert logger.disabled is False
        with silence_cmdstanpy():
            assert logger.disabled is True
            with silence_cmdstanpy():
                assert logger.disabled is True
            assert logger.disabled is True
        assert logger.disabled is False

    def test_logger_restored_after_an_error(self):
        with pytest.raises(RuntimeError):
            with silence_cmdstanpy():
                raise RuntimeError
        assert logging.getLogger("cmdstanpy").disabled is False

    def test_disabled_flag_leaves_logger_alone(self):
        with silence_cmdstanpy(False):
            assert logging.getLogger("cmdstanpy").disabled is False

    def test_decorator_follows_the_object_level(self):
        class _Thing(VerboseMixin):
            @quiet_when_silent
            def run(self):
                return logging.getLogger("cmdstanpy").disabled

        assert _Thing(verbose=0).run() is True
        assert _Thing(verbose=2).run() is False


class TestModelVerbosity:
    def test_integer_levels_on_the_constructor(self):
        assert StandardBKT(verbose=0).verbose is VerbosityLevel.SILENT
        assert StandardBKT(verbose=50).verbose is VerbosityLevel.DEBUG

    def test_fits_follow_the_model(self):
        model = StandardBKT(verbose=0)
        assert model.fits.verbose is VerbosityLevel.SILENT
        model.set_verbosity(100)
        assert model.verbose is VerbosityLevel.DEBUG
        assert model.fits.verbose is VerbosityLevel.DEBUG

    @pytest.mark.parametrize(
        "verbose, expected", [(0, False), (VerbosityLevel.WARN, True)]
    )
    def test_progress_bars_only_hidden_when_silent(self, verbose, expected):
        model = StandardBKT(verbose=verbose)
        seen = {}

        def _sample(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace()

        model._stan_model = SimpleNamespace(sample=_sample)
        model._fit_stan_model_using_method({}, MCMCFitOptions(show_progress=True))
        assert seen["show_progress"] is expected

    def test_methods_without_progress_bars_get_no_such_option(self):
        model = StandardBKT(fit_method=FitMethod.MLE, verbose=0)
        seen = {}

        def _optimize(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace()

        model._stan_model = SimpleNamespace(optimize=_optimize)
        model._fit_stan_model_using_method({}, MLEFitOptions())
        assert "show_progress" not in seen

    @pytest.mark.parametrize(
        "verbose, expected",
        [(0, VerbosityLevel.SILENT), (99, VerbosityLevel.DEBUG)],
    )
    def test_saved_level_loads(self, monkeypatch, tmp_path, verbose, expected):
        model = StandardBKT(verbose=verbose)
        monkeypatch.setattr(
            model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
        )

        def _fake(data_dict, fit_options):
            csv_file = tmp_path / "x.csv"
            csv_file.write_text("lp__\n0\n")
            return SimpleNamespace(runset=SimpleNamespace(csv_files=[str(csv_file)]))

        monkeypatch.setattr(model, "_fit_stan_model_using_method", _fake)
        monkeypatch.setattr(
            FitMethod,
            "infer_fit_method_from_stan_fit",
            staticmethod(lambda _: FitMethod.MCMC),
        )
        model.fit(
            pd.DataFrame(
                {
                    "student_id": ["s1", "s1"],
                    "problem_id": ["p1", "p2"],
                    "correct": [1, 0],
                    "timestamp": [1, 2],
                }
            )
        )
        archive = tmp_path / "model.stanbktmod"
        model.save(archive)

        with pytest.warns(UserWarning, match="Skipping this KC"):
            loaded = load_model(archive)
        assert loaded.verbose is expected
        assert loaded.fits.verbose is expected
