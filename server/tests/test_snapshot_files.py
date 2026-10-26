import errno
from types import SimpleNamespace

import pytest

from yizhu_server import snapshot_files as files


@pytest.fixture(autouse=True)
def enough_space(monkeypatch):
  monkeypatch.setattr(files.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))


def test_copy_is_independent_and_never_overwrites(tmp_path):
  source, target = tmp_path / "source", tmp_path / "target"
  source.write_bytes(b"original")
  files.copy_snapshot_file(source, target)
  assert target.stat().st_ino != source.stat().st_ino
  source.write_bytes(b"changed source")
  assert target.read_bytes() == b"original"
  with pytest.raises(FileExistsError):
    files.copy_snapshot_file(source, target)
  assert target.read_bytes() == b"original"


def test_reflink_failure_does_not_fallback_or_leave_a_partial_file(tmp_path, monkeypatch):
  source, target = tmp_path / "source", tmp_path / "target"
  source.write_bytes(b"original")

  def unsupported(*args):
    raise OSError(errno.EOPNOTSUPP, "Synthetic unsupported filesystem")

  monkeypatch.setattr(files.fcntl, "ioctl", unsupported)
  with pytest.raises(OSError):
    files.copy_snapshot_file(source, target, reflink=True)
  assert not target.exists()
  assert source.read_bytes() == b"original"


def test_requested_reflink_uses_clone_ioctl(tmp_path, monkeypatch):
  source, target = tmp_path / "source", tmp_path / "target"
  source.write_bytes(b"original")
  calls = []
  monkeypatch.setattr(files.fcntl, "ioctl", lambda *args: calls.append(args))
  files.copy_snapshot_file(source, target, reflink=True)
  assert len(calls) == 1 and calls[0][1] == files.FICLONE
  assert target.stat().st_ino != source.stat().st_ino


def test_insufficient_space_fails_before_creating_file(tmp_path, monkeypatch):
  source, target = tmp_path / "source", tmp_path / "target"
  source.write_bytes(b"original")
  monkeypatch.setattr(files.shutil, "disk_usage",
                      lambda _: SimpleNamespace(free=files.MINIMUM_FREE_BYTES))
  with pytest.raises(OSError, match="4 GiB"):
    files.copy_snapshot_file(source, target)
  assert not target.exists()
  with pytest.raises(OSError, match="4 GiB"):
    files.reserve_space(tmp_path)


def test_symlinks_are_not_snapshot_sources(tmp_path):
  source, link, target = tmp_path / "source", tmp_path / "link", tmp_path / "target"
  source.write_bytes(b"original")
  link.symlink_to(source)
  with pytest.raises(ValueError, match="symlink"):
    files.copy_snapshot_file(link, target)
  assert not target.exists()
