import pytest

from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel


class TestVerbosityLevel:
    def test_info_value(self):
        assert VerbosityLevel.INFO.value == 2

    def test_warn_value(self):
        assert VerbosityLevel.WARN.value == 1

    def test_debug_value(self):
        assert VerbosityLevel.DEBUG.value == 3

    def test_all_members_present(self):
        names = {m.name for m in VerbosityLevel}
        assert names == {"INFO", "WARN", "DEBUG"}

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

    def test_set_verbosity_raises_on_invalid_level(self):
        m = VerboseMixin()
        with pytest.raises(ValueError, match="Invalid verbosity level"):
            m.set_verbosity(999)  # type: ignore

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
