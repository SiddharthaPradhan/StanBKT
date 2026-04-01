from enum import IntEnum


class VerbosityLevel(IntEnum):
    WARN = 1
    INFO = 2
    DEBUG = 3


class VerboseMixin:
    def __init__(self, *, verbose: VerbosityLevel = VerbosityLevel.INFO, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def set_verbosity(self, level: VerbosityLevel):
        if level not in VerbosityLevel:
            raise ValueError(
                f"Invalid verbosity level: {level}. Must be one of {list(VerbosityLevel)}."
            )
        self.verbose = level

    def log(self, msg: str, level: VerbosityLevel = VerbosityLevel.INFO):
        if self.verbose >= level:
            if level == VerbosityLevel.WARN:
                msg = f"WARNING: {msg}"
            elif level == VerbosityLevel.DEBUG:
                msg = f"DEBUG: {msg}"
            print(msg, flush=True)
