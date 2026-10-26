from pathlib import Path

import pytest

from yizhu_server.target_model_metadata import build


def test_metadata_requires_existing_model_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "model", tmp_path / "features", tmp_path / "snapshot", "example")


def test_metadata_is_immutable(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        build(model, Path("features"), Path("snapshot"), "example")
