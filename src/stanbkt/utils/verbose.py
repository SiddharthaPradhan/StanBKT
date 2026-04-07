from enum import IntEnum


class VerbosityLevel(IntEnum):
    """Enumeration of verbosity levels for logging output.
    
    Attributes
    ----------
    WARN : int
        Warning-level messages only (highest verbosity threshold).
    INFO : int
        General information messages (default level).
    DEBUG : int
        Detailed debug-level messages (lowest verbosity threshold).
    """
    WARN = 1
    INFO = 2
    DEBUG = 3


class VerboseMixin:
    """Mixin class providing logging capabilities with verbosity control.
    
    This class adds logging methods to any class, allowing filtered output
    based on configured verbosity level.
    """
    def __init__(self, *, verbose: VerbosityLevel = VerbosityLevel.INFO, **kwargs):
        """Initialize verbosity mixin.
        
        Parameters
        ----------
        verbose : VerbosityLevel, default VerbosityLevel.INFO
            Initial verbosity level.
        **kwargs
            Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.verbose = verbose

    def set_verbosity(self, level: VerbosityLevel):
        """Set the verbosity level for logging.
        
        Parameters
        ----------
        level : VerbosityLevel
            New verbosity level.
        
        Raises
        ------
        ValueError
            If level is not a valid VerbosityLevel.
        """
        if level not in VerbosityLevel:
            raise ValueError(
                f"Invalid verbosity level: {level}. Must be one of {list(VerbosityLevel)}."
            )
        self.verbose = level

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
        if self.verbose >= level:
            if level == VerbosityLevel.WARN:
                msg = f"WARNING: {msg}"
            elif level == VerbosityLevel.DEBUG:
                msg = f"DEBUG: {msg}"
            print(msg, flush=True)
