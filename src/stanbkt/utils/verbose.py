from enum import Enum


class VerbosityLevel(Enum):
    INFO = 1
    WARN = 2
    DEBUG = 3


class VerboseMixin:
    def __init__(self, *, verbose: VerbosityLevel = VerbosityLevel.INFO, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def _print(self, msg: str, level: VerbosityLevel = VerbosityLevel.INFO):
        if self.verbose.value >= level.value:
            if level == VerbosityLevel.WARN:
                msg = f"WARNING: {msg}"
            elif level == VerbosityLevel.DEBUG:
                msg = f"DEBUG: {msg}"
            print(msg, flush=True)
