"""Temporary work folders for CmdStan output and evicted fits."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from platformdirs import PlatformDirs

if os.name == "nt":
    import msvcrt
else:
    import fcntl


WORK_DIR_PREFIX = "stanbkt_work_"
LOCK_FILE = ".lock"
# a new work dir is not locked yet for a brief moment, so young folders are never swept
SWEEP_GRACE_SECONDS = 60.0
_RAM_FILESYSTEMS = {"tmpfs", "ramfs"}

# keeps a work dir alive for as long as a fit or CmdStanGQ that reads from it
_KEEP_ALIVE: weakref.WeakKeyDictionary[Any, "WorkDir"] = weakref.WeakKeyDictionary()
_KEEP_ALIVE_LOCK = threading.Lock()


def default_cache_root() -> Path:
    """Return the temp folder in the user cache, separate from the compiled model cache."""
    return PlatformDirs(appname="stanbkt", appauthor=False).user_cache_path / "tmp"


def resolve_temp_root(
    temp_dir: str | os.PathLike[str] | None = None, low_memory: bool = False
) -> str:
    """Return the folder that holds work dirs, creating it if needed.

    Uses ``temp_dir``, then ``TMPDIR``. Otherwise ``low_memory`` uses the user cache, because
    the system temp dir is often held in RAM, and other models use the system temp dir.

    Parameters
    ----------
    temp_dir : str | os.PathLike[str] | None, optional
        Folder chosen by the user.
    low_memory : bool, default=False
        Whether the model keeps its fits on disk.

    Returns
    -------
    str
        Absolute path of a writable folder.

    Raises
    ------
    ValueError
        If the folder cannot be created or written to.
    """
    if temp_dir is not None:
        root = os.path.expanduser(os.fspath(temp_dir))
    elif os.environ.get("TMPDIR"):
        root = os.environ["TMPDIR"]
    elif low_memory:
        root = os.fspath(default_cache_root())
    else:
        root = tempfile.gettempdir()

    try:
        os.makedirs(root, exist_ok=True)
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise ValueError(f"Cannot write temporary files to '{root}': {exc}") from exc
    return os.path.abspath(root)


def _filesystem_type(path: str, mounts_file: str = "/proc/self/mounts") -> str | None:
    """Return the filesystem type of ``path``, None where the mount table cannot be read."""
    resolved = os.path.realpath(path)
    best_mount, best_type = "", None
    try:
        with open(mounts_file, "r", encoding="utf-8", errors="replace") as mounts:
            for line in mounts:
                fields = line.split()
                if len(fields) < 3:
                    continue
                # the mount table writes spaces in paths as octal escapes
                mount_point = fields[1].replace("\\040", " ")
                prefix = mount_point.rstrip("/") + "/"
                if resolved != mount_point and not resolved.startswith(prefix):
                    continue
                # the last entry wins for mounts stacked on the same path
                if len(mount_point) >= len(best_mount):
                    best_mount, best_type = mount_point, fields[2]
    except OSError:
        return None
    return best_type


def ram_temp_notice(root: str, temp_dir: str | os.PathLike[str] | None) -> str | None:
    """Return a message when temporary files were sent to a RAM backed folder by default.

    Parameters
    ----------
    root : str
        Folder returned by :func:`resolve_temp_root`.
    temp_dir : str | os.PathLike[str] | None
        Folder chosen by the user, which never gets a message.

    Returns
    -------
    str | None
        The message, or None when the folder is on disk, was chosen with ``temp_dir``, or its
        filesystem cannot be determined.
    """
    if temp_dir is not None or _filesystem_type(root) not in _RAM_FILESYSTEMS:
        return None
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return None
    return (
        f"Temporary files are going to '{root}', which is held in RAM, {usage.free / 1e9:.1f} GB "
        f"free of {usage.total / 1e9:.1f} GB. Pass 'temp_dir' to use a folder on disk."
    )


def _try_lock(fd: int) -> bool:
    """Try to take an exclusive lock without waiting. The OS releases it when the process dies."""
    try:
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _remove_work_dir(path: str) -> None:
    """Remove a work dir, keeping the lock file until last so a partial removal is swept again."""
    try:
        entries = list(os.scandir(path))
    except OSError:
        return
    for entry in entries:
        if entry.name == LOCK_FILE:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            try:
                os.remove(entry.path)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _cleanup_work_dir(path: str, lock_fd: int, owner_pid: int) -> None:
    # a forked child inherits the finalizer, only the process that created the folder removes it
    if os.getpid() != owner_pid:
        return
    try:
        os.close(lock_fd)
    except OSError:
        pass
    _remove_work_dir(path)


def sweep_stale_work_dirs(root: str | os.PathLike[str]) -> list[str]:
    """Remove work dirs left behind by processes that did not exit cleanly.

    Only folders named like a work dir that contain a lock file are considered. A folder is
    removed when its lock can be taken, which means its owner is gone. Folders in the grace
    period and folders whose lock cannot be checked are left alone.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder to look in.

    Returns
    -------
    list[str]
        Paths of the removed work dirs.
    """
    removed: list[str] = []
    try:
        entries = list(os.scandir(root))
    except OSError:
        return removed

    for entry in entries:
        try:
            if not entry.name.startswith(WORK_DIR_PREFIX) or not entry.is_dir(
                follow_symlinks=False
            ):
                continue
            lock_path = os.path.join(entry.path, LOCK_FILE)
            if time.time() - os.path.getmtime(lock_path) < SWEEP_GRACE_SECONDS:
                continue
            fd = os.open(lock_path, os.O_RDWR)
        except OSError:
            continue
        try:
            owner_is_gone = _try_lock(fd)
        finally:
            os.close(fd)
        if owner_is_gone:
            _remove_work_dir(entry.path)
            removed.append(entry.path)
    return removed


def start_background_sweep() -> threading.Thread | None:
    """Sweep the default cache root on a daemon thread, so that importing stays fast."""
    root = default_cache_root()
    if not root.is_dir():
        return None
    thread = threading.Thread(
        target=sweep_stale_work_dirs, args=(root,), name="stanbkt-temp-sweep", daemon=True
    )
    thread.start()
    return thread


def run_dir_of(stan_output: Any) -> str | None:
    """Return the folder holding the CSVs of a CmdStan fit or CmdStanGQ, if known."""
    runset = getattr(stan_output, "runset", None)
    csv_files = getattr(runset, "csv_files", None)
    if not isinstance(csv_files, (list, tuple)) or not csv_files:
        return None
    return os.path.dirname(os.path.abspath(os.fspath(csv_files[0])))


class WorkDir:
    """A folder owned by one model, removed on exit or by a later sweep after a crash.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder to create the work dir in.
    """

    def __init__(self, root: str | os.PathLike[str]):
        sweep_stale_work_dirs(root)
        self.path: str = tempfile.mkdtemp(prefix=WORK_DIR_PREFIX, dir=os.fspath(root))
        lock_fd = os.open(
            os.path.join(self.path, LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600
        )
        _try_lock(lock_fd)
        self._finalizer = weakref.finalize(
            self, _cleanup_work_dir, self.path, lock_fd, os.getpid()
        )

    def new_run_dir(self, prefix: str) -> str:
        """Create a unique folder for one CmdStan run, as CmdStan file names can collide."""
        return tempfile.mkdtemp(prefix=f"{prefix}_", dir=self.path)

    def remove_run_dir(self, path: str | None) -> None:
        """Remove a run folder. Paths outside this work dir are ignored."""
        if path is None:
            return
        resolved = os.path.abspath(path)
        if os.path.dirname(resolved) == os.path.abspath(self.path):
            shutil.rmtree(resolved, ignore_errors=True)

    def remove_run_dir_with(self, stan_output: Any, path: str) -> None:
        """Remove a run folder once ``stan_output`` is garbage collected."""
        try:
            weakref.finalize(
                stan_output, shutil.rmtree, path, ignore_errors=True
            ).atexit = False
        except TypeError:
            # not weakly referenceable, only test doubles
            pass

    def keep_alive(self, stan_output: Any) -> None:
        """Keep this folder for as long as ``stan_output`` exists, even if the model is deleted."""
        try:
            with _KEEP_ALIVE_LOCK:
                _KEEP_ALIVE[stan_output] = self
        except TypeError:
            # not hashable or not weakly referenceable, only test doubles
            pass

    def cleanup(self) -> None:
        """Remove the folder now."""
        self._finalizer()
