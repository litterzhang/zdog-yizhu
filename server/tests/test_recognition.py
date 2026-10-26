"""Offline provider-boundary tests: no model downloads, network or real weights."""

import hashlib
import json

import pytest

from yizhu_server import recognition
from yizhu_server.recognition import (
    BioClipProvider,
    Label,
    format_candidates,
    load_labels,
    read_photo,
    verify_weights,
)


def labels_file(tmp_path, rows, version="test-v1"):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"version": version, "labels": rows}))
    return path


def test_default_labels_are_versioned_unique_and_include_nonplant_controls():
    version, labels = load_labels()
    assert version == "baseline-labels-v2"
    assert len(labels) == 29
    assert len({label.id for label in labels}) == 29
    assert sum(label.kind == "nonplant" for label in labels) == 5
    raw = json.loads(recognition.DEFAULT_LABELS.read_text())
    assert all("common_name" not in row for row in raw["labels"])


@pytest.mark.parametrize("rows", [
    [],
    [{"id": "one", "name": "Ginkgo biloba", "kind": "plant"}],
    [{"id": "one", "name": "Ginkgo biloba", "kind": "plant"}] * 2,
    [
        {"id": "one", "name": "Ginkgo biloba", "kind": "plant"},
        {"id": "two", "name": "Ginkgo biloba", "kind": "plant"},
    ],
    [
        {"id": "one", "name": "", "kind": "plant"},
        {"id": "two", "name": "car", "kind": "nonplant"},
    ],
    [None, {}],
])
def test_invalid_label_sets_are_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        load_labels(labels_file(tmp_path, rows))


def test_weights_are_size_and_sha_checked_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.bin"
    path.write_bytes(b"abc")
    monkeypatch.setattr(recognition, "MODEL_BYTES", 3)
    monkeypatch.setattr(recognition, "MODEL_SHA256", hashlib.sha256(b"abc").hexdigest())
    verify_weights(path)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum"):
        verify_weights(path)
    path.write_bytes(b"too long")
    with pytest.raises(ValueError, match="size"):
        verify_weights(path)
    with pytest.raises(ValueError):
        verify_weights(tmp_path / "missing.bin")


def test_high_softmax_is_not_a_confidence_or_confirmed_species():
    result = format_candidates(
        [{"classification": "Ginkgo biloba", "score": 0.999}],
        [Label("local-label", "plant", "Ginkgo biloba")],
        label_version="test-v1",
    )
    assert result["is_demo"] is False
    assert result["status"] == "requires_review"
    assert result["candidates"][0]["confidence"] is None
    assert result["candidates"][0]["score"] == 0.999
    assert result["revision"] == recognition.MODEL_REVISION
    assert "species_id" not in result["candidates"][0]


def test_nonplant_controls_are_not_formatted_as_species():
    result = format_candidates(
        [{"classification": "a car", "score": 0.8}],
        [Label("control", "nonplant", "a car")], label_version="test-v1",
    )
    assert result["candidates"][0]["scientific_name"] is None
    assert result["status"] == "requires_review"


@pytest.mark.parametrize("rows", [
    [],
    [{"classification": "not-in-label-set", "score": 0.9}],
    [{"classification": "Ginkgo biloba", "score": float("nan")}],
    [{"classification": "Ginkgo biloba", "score": 1.1}],
    [{"classification": "Ginkgo biloba", "score": True}],
])
def test_malformed_inference_output_fails_closed(rows):
    with pytest.raises(ValueError):
        format_candidates(rows, [Label("one", "plant", "Ginkgo biloba")], label_version="v1")


def test_provider_uses_injected_ranker_without_model_or_network():
    class Ranker:
        def predict(self, images, **kwargs):
            assert images == ["in-memory-test"]
            assert kwargs["k"] == 1
            return [{"classification": "Ginkgo biloba", "score": 0.6}]

    provider = BioClipProvider.__new__(BioClipProvider)
    provider.labels = [Label("one", "plant", "Ginkgo biloba")]
    provider.label_version = "test"
    provider.classifier = Ranker()
    assert provider.recognize("in-memory-test", top_k=1)["candidates"][0]["score"] == 0.6
    with pytest.raises(ValueError):
        provider.recognize("in-memory-test", top_k=2)


def test_photo_orientation_and_metadata_are_handled_before_inference(tmp_path):
    Image = pytest.importorskip("PIL.Image", reason="optional recognition extra")
    image = Image.new("RGB", (12, 8), "green")
    exif = Image.Exif()
    exif[274] = 6  # Rotate clockwise.
    exif[315] = "private metadata"
    path = tmp_path / "photo.jpg"
    image.save(path, exif=exif)
    with read_photo(path) as clean:
        assert clean.size == (8, 12)
        assert clean.mode == "RGB"
        assert clean.info == {}
        assert len(clean.getexif()) == 0


def test_invalid_or_oversized_images_are_rejected(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image", reason="optional recognition extra")
    path = tmp_path / "image.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        read_photo(path)
    Image.new("RGB", (3, 3)).save(path)
    monkeypatch.setattr(recognition, "MAX_IMAGE_PIXELS", 4)
    with pytest.raises(ValueError, match="pixel"):
        read_photo(path)
    monkeypatch.setattr(recognition, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="8 MiB"):
        read_photo(path)


def test_unsupported_and_animated_formats_are_rejected(tmp_path):
    Image = pytest.importorskip("PIL.Image", reason="optional recognition extra")
    path = tmp_path / "image.bmp"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError, match="Only"):
        read_photo(path)
    animated = tmp_path / "animated.png"
    Image.new("RGB", (2, 2), "white").save(
        animated, save_all=True, append_images=[Image.new("RGB", (2, 2), "black")], duration=100,
    )
    with pytest.raises(ValueError, match="Animated"):
        read_photo(animated)


def test_cli_refuses_to_overwrite_existing_report_before_loading_model(tmp_path, monkeypatch):
    from yizhu_server import recognition_cli

    report = tmp_path / "existing.json"
    report.write_text("keep me")
    monkeypatch.setattr("sys.argv", ["yizhu-recognize", "not-used.jpg", "--output", str(report)])
    monkeypatch.setattr(recognition_cli, "BioClipProvider", lambda *_args, **_kwargs: pytest.fail(
        "No model should load before the overwrite check",
    ))
    with pytest.raises(SystemExit) as error:
        recognition_cli.main()
    assert error.value.code == 1
    assert report.read_text() == "keep me"


def test_api_import_does_not_import_inference_dependencies():
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from yizhu_server.app import create_app; "
        "assert 'torch' not in sys.modules; assert 'bioclip' not in sys.modules",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0
