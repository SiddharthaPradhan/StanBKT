import pytest

from stanbkt.utils import temp_dirs


@pytest.fixture(autouse=True)
def _isolated_temp_root(monkeypatch, tmp_path_factory):
    # models write work dirs here, never to the real user cache
    monkeypatch.setenv("TMPDIR", str(tmp_path_factory.mktemp("stanbkt_tmp")))
    # the outcome must not depend on whether the machine has a RAM backed /tmp
    monkeypatch.setattr(temp_dirs, "_filesystem_type", lambda path: "ext4")
