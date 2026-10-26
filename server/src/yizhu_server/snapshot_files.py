"""Independent public-data snapshots with an explicit Linux copy-on-write option."""

import fcntl
import shutil
import stat

MINIMUM_FREE_BYTES = 4 * 1024**3
FICLONE = 0x40049409  # Linux include/uapi/linux/fs.h: _IOW(0x94, 9, int).


def reserve_space(directory, required_bytes=64 * 1024):
  if shutil.disk_usage(directory).free < MINIMUM_FREE_BYTES + required_bytes:
    raise OSError("Keep at least 4 GiB free for collection and training.")


def copy_snapshot_file(source, destination, *, reflink=False):
  """Never hard-link or overwrite. Explicit reflink failure must not trigger a large copy."""
  info = source.lstat()
  if not stat.S_ISREG(info.st_mode):
    raise ValueError("Snapshot source must be a regular file, not a symlink.")
  reserve_space(destination.parent, 64 * 1024 if reflink else info.st_size)
  created = False
  try:
    with source.open("rb") as reader, destination.open("xb") as writer:
      created = True
      if reflink:
        fcntl.ioctl(writer.fileno(), FICLONE, reader.fileno())
      else:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    reserve_space(destination.parent, 0)
  except BaseException:
    if created:
      destination.unlink(missing_ok=True)
    raise
