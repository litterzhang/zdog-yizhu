"""Atomic import of verified remote CPU features, without replacing local results."""

import os
import stat
import tempfile
from pathlib import Path

import numpy as np

from yizhu_server.snapshot_files import copy_snapshot_file
from yizhu_server.training_data import sha256

DIMENSION = 2048


def read_feature(path, *, normalized):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size < DIMENSION * 4 + 1024:
        raise ValueError("Invalid feature file: expected a small regular file, not a symlink.")
    vector = np.load(path, allow_pickle=False)
    if vector.shape != (DIMENSION,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError("Invalid feature shape, dtype or values.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0 or (normalized and not np.isclose(norm, 1, atol=.001)):
        raise ValueError("Invalid feature norm.")
    return vector


def publish_feature(source, destination, *, expected_sha256, normalized):
    """Caller verifies snapshot/encoder identity; publish each complete array without clobbering."""
    reference = read_feature(source, normalized=normalized)
    if sha256(source) != expected_sha256:
        raise ValueError("Remote feature checksum mismatch.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def preserve_existing():
        existing = read_feature(destination, normalized=normalized)
        difference = float(np.max(np.abs(existing - reference)))
        if not np.allclose(existing, reference, atol=1e-5, rtol=1e-4):
            raise ValueError("Existing local feature differs; preserve it and inspect.")
        return {"status": "preserved-existing", "maximumAbsoluteDifference": difference}

    if destination.exists() or destination.is_symlink():
        return preserve_existing()
    # A complete private temporary copy is atomically linked into the final name.
    # This never links the incoming staging file itself; after cleanup nlink is 1.
    with tempfile.TemporaryDirectory(prefix=".feature-import-", dir=destination.parent) as work:
        temporary = Path(work) / "complete.npy"
        copy_snapshot_file(source, temporary, reflink=True)
        if sha256(temporary) != expected_sha256:
            raise ValueError("Feature changed during import.")
        read_feature(temporary, normalized=normalized)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return preserve_existing()
    return {"status": "imported", "maximumAbsoluteDifference": 0.0}
