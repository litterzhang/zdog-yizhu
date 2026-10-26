import shutil

import numpy as np
import pytest

from yizhu_server import feature_exchange as exchange
from yizhu_server.training_data import sha256


@pytest.fixture
def feature_files(tmp_path, monkeypatch):
    # The actual Btrfs reflink primitive has separate tests; /tmp can be tmpfs.
    def copy(source, destination, *, reflink):
        assert reflink is True
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)

    monkeypatch.setattr(exchange, "copy_snapshot_file", copy)
    source, destination = tmp_path / "incoming.npy", tmp_path / "cache/feature.npy"
    vector = np.ones(2048, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    np.save(source, vector, allow_pickle=False)
    return source, destination, vector


def test_import_is_independent_and_final_name_is_only_published_when_complete(
    feature_files, monkeypatch,
):
    source, destination, vector = feature_files
    link = exchange.os.link

    def inspect_link(temporary, target):
        assert not target.exists()
        np.testing.assert_array_equal(np.load(temporary, allow_pickle=False), vector)
        assert temporary.stat().st_ino != source.stat().st_ino
        link(temporary, target)

    monkeypatch.setattr(exchange.os, "link", inspect_link)
    result = exchange.publish_feature(
        source, destination, expected_sha256=sha256(source), normalized=True,
    )
    assert result["status"] == "imported"
    assert destination.stat().st_nlink == 1
    assert source.stat().st_ino != destination.stat().st_ino
    assert not list(destination.parent.glob(".feature-import-*"))


@pytest.mark.parametrize("racing", [False, True])
def test_existing_result_is_not_overwritten_even_if_created_during_import(
    feature_files, monkeypatch, racing,
):
    source, destination, vector = feature_files
    existing = vector.copy()
    existing[0] += 1e-7
    destination.parent.mkdir()
    if racing:
        link = exchange.os.link

        def race(temporary, target):
            np.save(target, existing, allow_pickle=False)
            link(temporary, target)

        monkeypatch.setattr(exchange.os, "link", race)
    else:
        np.save(destination, existing, allow_pickle=False)
    result = exchange.publish_feature(
        source, destination, expected_sha256=sha256(source), normalized=True,
    )
    assert result["status"] == "preserved-existing"
    np.testing.assert_array_equal(np.load(destination, allow_pickle=False), existing)


def test_checksum_mismatch_does_not_publish(feature_files):
    source, destination, _ = feature_files
    with pytest.raises(ValueError, match="checksum"):
        exchange.publish_feature(source, destination, expected_sha256="bad", normalized=True)
    assert not destination.exists()


def test_disagreement_preserves_local_result(feature_files):
    source, destination, vector = feature_files
    destination.parent.mkdir()
    np.save(destination, -vector, allow_pickle=False)
    digest = sha256(destination)
    with pytest.raises(ValueError, match="differs"):
        exchange.publish_feature(source, destination, expected_sha256=sha256(source), normalized=True)
    assert sha256(destination) == digest


@pytest.mark.parametrize("bad", [
    np.zeros(2048, dtype=np.float32),
    np.ones(2048, dtype=np.float64),
    np.ones(2047, dtype=np.float32),
    np.full(2048, np.nan, dtype=np.float32),
    np.full(2048, np.inf, dtype=np.float32),
    np.ones(2048, dtype=np.float32),
])
def test_invalid_normalized_arrays_rejected(feature_files, bad):
    source, destination, _ = feature_files
    np.save(source, bad, allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid feature"):
        exchange.publish_feature(source, destination, expected_sha256=sha256(source), normalized=True)
    assert not destination.exists()


def test_raw_species_features_do_not_require_unit_norm(feature_files):
    source, destination, _ = feature_files
    np.save(source, np.ones(2048, dtype=np.float32), allow_pickle=False)
    assert exchange.publish_feature(
        source, destination, expected_sha256=sha256(source), normalized=False,
    )["status"] == "imported"


def test_symlink_destination_rejected_without_touching_target(feature_files):
    source, destination, _ = feature_files
    destination.parent.mkdir()
    destination.symlink_to(source)
    digest = sha256(source)
    with pytest.raises(ValueError, match="symlink"):
        exchange.publish_feature(source, destination, expected_sha256=digest, normalized=True)
    assert sha256(source) == digest and destination.is_symlink()


def test_clone_failure_leaves_no_partial_final_file(feature_files, monkeypatch):
    source, destination, _ = feature_files

    def fail(*args, **kwargs):
        raise OSError("Synthetic full disk")

    monkeypatch.setattr(exchange, "copy_snapshot_file", fail)
    with pytest.raises(OSError, match="full disk"):
        exchange.publish_feature(source, destination, expected_sha256=sha256(source), normalized=True)
    assert not destination.exists()
    assert not list(destination.parent.glob(".feature-import-*"))
