import hashlib
import json
import random

import pytest
from PIL import Image

from yizhu_server import scratch_presence_training as scratch
from yizhu_server.plant_presence import load_dataset


def plan():
    return {
        "task": "plant-presence", "initialization": "random", "labels": ["nonplant", "plant"],
        "seed": 20260925, "pretrainedWeightsAllowed": False, "pretrainedFeaturesAllowed": False,
        "testImagesIncluded": False, "automaticDeployment": False,
        "samples": [{"split": "train", "label": "nonplant", "labelEvidence": {"placeholder": True}}],
    }


@pytest.mark.parametrize("change", [
    {"pretrainedWeightsAllowed": True}, {"pretrainedWeightsAllowed": 0},
    {"pretrainedFeaturesAllowed": None}, {"testImagesIncluded": True},
    {"automaticDeployment": "false"}, {"labels": ["plant", "nonplant"]},
    {"initialization": "imagenet"}, {"seed": True}, {"seed": -1},
    {"samples": []}, {"samples": None},
])
def test_clean_start_plan_rejects_ambiguous_or_wrong_settings(change):
    with pytest.raises(ValueError):
        scratch.validate_plan({**plan(), **change})


def test_plan_requires_explicit_label_evidence_and_no_test_rows():
    scratch.validate_plan(plan())
    for change in [{"split": "test"}, {"labelEvidence": None}, {"label": "unknown"}]:
        document = plan()
        document["samples"][0].update(change)
        with pytest.raises(ValueError):
            scratch.validate_plan(document)


@pytest.fixture
def raw_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    rows = []
    for index, (split, label) in enumerate([
        ("train", "nonplant"), ("train", "plant"),
        ("validation", "nonplant"), ("validation", "plant"),
    ]):
        path = tmp_path / "images" / f"{index}.png"
        Image.frombytes("RGB", (32, 32), random.Random(index + 100).randbytes(3072)).save(path)
        rows.append({
            "id": str(index), "label": label, "split": split,
            "path": f"images/{index}.png", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "group": f"synthetic-author-{index}", "license": "cc0",
            "attribution": "Synthetic unit-test fixture", "sourceUrl": "https://example.org/fixture",
            "public": True, "reviewed": True,
            "labelEvidence": {
                "dataset": "Open Images V7", "classId": "/m/05s2s",
                "confidence": int(label == "plant"), "sources": ["verification"],
            },
        })
    (tmp_path / "samples.json").write_text(json.dumps({"version": 1, "samples": rows}))
    return tmp_path, rows


def test_training_loader_accepts_only_requested_splits(raw_dataset):
    root, rows = raw_dataset
    assert len(load_dataset(root, required_splits=("train", "validation"))[0]) == 4
    with pytest.raises(ValueError):
        load_dataset(root)
    rows[2]["split"] = "test"
    (root / "samples.json").write_text(json.dumps({"version": 1, "samples": rows}))
    with pytest.raises(ValueError):
        load_dataset(root, required_splits=("train", "validation"))


@pytest.mark.parametrize("change", [
    {"confidence": 1}, {"sources": ["machine"]}, {"sources": []},
    {"dataset": "unknown"}, {"classId": "a-nonplant-category"},
])
def test_negative_requires_explicit_human_absence_label(raw_dataset, change):
    root, rows = raw_dataset
    rows[0]["labelEvidence"].update(change)
    (root / "samples.json").write_text(json.dumps({"version": 1, "samples": rows}))
    with pytest.raises(ValueError, match="contradicts"):
        load_dataset(root, required_splits=("train", "validation"))


def test_pause_is_honored_before_model_creation(tmp_path, monkeypatch):
    (tmp_path / "paused.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("No model should be initialized for paused data")

    monkeypatch.setattr(scratch, "build_random_classifier", forbidden)
    with pytest.raises(ValueError, match="paused"):
        scratch.train(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_whole_frame_preserves_both_edges_instead_of_center_cropping():
    image = Image.new("RGB", (320, 160), "black")
    image.paste("red", (0, 0, 40, 160))
    image.paste("green", (280, 0, 320, 160))
    output = scratch.full_frame(image, 160)
    assert output.getpixel((5, 80)) == (255, 0, 0)
    assert output.getpixel((155, 80)) == (0, 128, 0)
    assert output.getpixel((80, 0)) == (128, 128, 128)
