from contextlib import contextmanager
from enum import IntEnum
import functools
import logging
from numbers import Integral
import threading


class VerbosityLevel(IntEnum):
    """Enumeration of verbosity levels for logging output.

    Integers are accepted as well. Values above ``DEBUG`` are treated as ``DEBUG``.

    Attributes
    ----------
    SILENT : int
        No logging, including CmdStanPy's log messages and progress bars.
    WARN : int
        Warning-level messages only (highest verbosity threshold).
    INFO : int
        General information messages (default level).
    DEBUG : int
        Detailed debug-level messages (lowest verbosity threshold).
    """
    SILENT = 0
    WARN = 1
    INFO = 2
    DEBUG = 3

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, Integral) and value > cls.DEBUG:
            return cls.DEBUG
        return None


def _resolve_level(level: VerbosityLevel | int) -> VerbosityLevel:
    try:
        return VerbosityLevel(level)
    except ValueError:
        raise ValueError(
            f"Invalid verbosity level: {level}. Must be one of {list(VerbosityLevel)} "
            "or an integer above them."
        ) from None


_cmdstanpy_logger = logging.getLogger("cmdstanpy")
_silenced_lock = threading.Lock()
_silenced_count = 0
_cmdstanpy_was_disabled = False


@contextmanager
def silence_cmdstanpy(enabled: bool = True):
    """Disable CmdStanPy's logger while the block runs.

    The logger is shared by the whole process, so overlapping blocks are counted and the
    logger is restored when the last one ends.

    Parameters
    ----------
    enabled : bool, default True
        Whether to silence the logger, ``False`` leaves it unchanged.
    """
    global _silenced_count, _cmdstanpy_was_disabled
    if not enabled:
        yield
        return
    with _silenced_lock:
        if _silenced_count == 0:
            _cmdstanpy_was_disabled = _cmdstanpy_logger.disabled
            _cmdstanpy_logger.disabled = True
        _silenced_count += 1
    try:
        yield
    finally:
        with _silenced_lock:
            _silenced_count -= 1
            if _silenced_count == 0:
                _cmdstanpy_logger.disabled = _cmdstanpy_was_disabled


def quiet_when_silent(method):
    """Decorate a method so that CmdStanPy does not log while its object is ``SILENT``."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with silence_cmdstanpy(self.verbose == VerbosityLevel.SILENT):
            return method(self, *args, **kwargs)

    return wrapper


class VerboseMixin:
    """Mixin class providing logging capabilities with verbosity control.

    This class adds logging methods to any class, allowing filtered output
    based on configured verbosity level.
    """
    def __init__(self, *, verbose: VerbosityLevel | int = VerbosityLevel.INFO, **kwargs):
        """Initialize verbosity mixin.

        Parameters
        ----------
        verbose : VerbosityLevel, default VerbosityLevel.INFO
            Initial verbosity level. ``0`` disables logging and values above ``3`` are
            treated as ``3``.
        **kwargs
            Additional keyword arguments passed to parent class.

        Raises
        ------
        ValueError
            If verbose is not a valid verbosity level.
        """
        super().__init__(**kwargs)
        self.verbose = _resolve_level(verbose)

    def set_verbosity(self, level: VerbosityLevel | int):
        """Set the verbosity level for logging.

        Parameters
        ----------
        level : VerbosityLevel
            New verbosity level. ``0`` disables logging and values above ``3`` are
            treated as ``3``.

        Raises
        ------
        ValueError
            If level is not a valid VerbosityLevel.
        """
        self.verbose = _resolve_level(level)

    def log(self, msg: str, level: VerbosityLevel = VerbosityLevel.INFO):
        """Log a message if verbosity level permits.

        Parameters
        ----------
        msg : str
            Message to log.
        level : VerbosityLevel, default VerbosityLevel.INFO
            Verbosity level of this message. Message is printed if
            self.verbose >= level. Lower enum values = higher verbosity.
        """
        if self.verbose != VerbosityLevel.SILENT and self.verbose >= level:
            if level == VerbosityLevel.WARN:
                msg = f"WARNING: {msg}"
            elif level == VerbosityLevel.DEBUG:
                msg = f"DEBUG: {msg}"
            print(msg, flush=True)
