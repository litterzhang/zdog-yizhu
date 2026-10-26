import hashlib
import json
import random
from copy import deepcopy

import pytest
from PIL import Image

from yizhu_server import plant_presence as presence


@pytest.fixture
def dataset(tmp_path):
  (tmp_path / "images").mkdir()
  rows = []
  for index, (split, label) in enumerate(
    (split, label) for split in presence.SPLITS for label in presence.LABELS
  ):
    path = tmp_path / "images" / f"{index}.png"
    image = Image.frombytes("RGB", (32, 32), random.Random(index).randbytes(32 * 32 * 3))
    image.save(path)
    rows.append({
      "id": str(index), "path": f"images/{index}.png", "label": label, "split": split,
      "group": f"group-{index}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "license": "cc0", "attribution": "Synthetic test fixture, not training evidence",
      "sourceUrl": "https://example.org/test-fixture", "public": True, "reviewed": True,
    })
  (tmp_path / "samples.json").write_text(json.dumps({"version": 1, "samples": rows}))
  return tmp_path, rows


def rewrite(root, rows):
  (root / "samples.json").write_text(json.dumps({"version": 1, "samples": rows}))


def test_dataset_requires_both_classes_in_each_split(dataset):
  root, rows = dataset
  assert len(presence.load_dataset(root)[0]) == 6
  rows[-1]["label"] = "nonplant"
  rewrite(root, rows)
  with pytest.raises(ValueError, match="both"):
    presence.load_dataset(root)


@pytest.mark.parametrize(("key", "value"), [
  ("reviewed", False), ("public", False), ("license", "cc-by-nc"),
  ("label", "animal"), ("label", "uncertain"), ("sourceUrl", "file:///tmp/image"),
])
def test_unreviewed_private_or_nonbinary_records_rejected(dataset, key, value):
  root, rows = dataset
  rows[0][key] = value
  rewrite(root, rows)
  with pytest.raises(ValueError):
    presence.load_dataset(root)


def test_photographer_group_must_not_leak(dataset):
  root, rows = dataset
  rows[-1]["group"] = rows[0]["group"]
  rewrite(root, rows)
  with pytest.raises(ValueError, match="leaks"):
    presence.load_dataset(root)


@pytest.mark.parametrize("change", ["wrong-label", "machine", "missing-sources"])
def test_dataset_human_verification_cannot_be_overridden(dataset, change):
  root, rows = dataset
  evidence = {
    "dataset": "Open Images V7", "classId": "/m/05s2s", "confidence": 0,
    "sources": ["verification"],
  }
  if change == "wrong-label":
    evidence["confidence"] = 1
  elif change == "machine":
    evidence["sources"] = ["machine"]
  else:
    evidence["sources"] = []
  rows[0]["labelEvidence"] = evidence
  rewrite(root, rows)
  with pytest.raises(ValueError, match="contradicts"):
    presence.load_dataset(root)


@pytest.mark.parametrize("mutation", ["duplicate", "checksum", "escape", "symlink"])
def test_invalid_images_rejected(dataset, mutation, tmp_path):
  root, rows = dataset
  if mutation == "duplicate":
    rows[-1].update({key: rows[0][key] for key in ("path", "sha256")})
  elif mutation == "checksum":
    rows[0]["sha256"] = "0" * 64
  elif mutation == "escape":
    rows[0]["path"] = "../outside.png"
  else:
    outside = root / "outside.png"
    outside.write_bytes((root / rows[0]["path"]).read_bytes())
    link = root / "images" / "link.png"
    link.symlink_to(outside)
    rows[0]["path"] = "images/link.png"
  rewrite(root, rows)
  with pytest.raises(ValueError):
    presence.load_dataset(root)


def test_letterbox_retains_both_edges():
  image = Image.new("RGB", (448, 224), "black")
  image.paste("red", (0, 0, 80, 224))
  image.paste("green", (368, 0, 448, 224))
  with presence.whole_image(image) as output:
    assert output.size == (224, 224)
    assert output.getpixel((5, 112)) == (255, 0, 0)
    assert output.getpixel((218, 112)) == (0, 128, 0)
    assert output.getpixel((112, 0)) == (124, 116, 104)


def test_binary_metrics_report_false_acceptance_not_top5():
  metrics = presence.binary_metrics([0.9, 0.7, 0.2, 0.1], [1, 0, 1, 0], 0.5)
  assert metrics["truePositive"] == metrics["falsePositive"] == 1
  assert metrics["trueNegative"] == metrics["falseNegative"] == 1
  assert metrics["plantRecall"] == metrics["nonplantRecall"] == 0.5
  assert metrics["falsePositiveRate"] == 0.5
  assert not any("top" in key.lower() for key in metrics)


@pytest.mark.parametrize("scores,labels,threshold", [
  ([], [], 0.5), ([float("nan")], [1], 0.5), ([1.1], [0], 0.5),
  ([0.5], [], 0.5), ([0.1], [2], 0.5), ([0.1], [0], float("nan")),
])
def test_invalid_metrics(scores, labels, threshold):
  with pytest.raises(ValueError):
    presence.binary_metrics(scores, labels, threshold)


def test_threshold_uses_validation_positive_quantile():
  scores, labels = [0.9, 0.8, 0.6, 0.1], [1, 1, 1, 0]
  threshold = presence.select_threshold(scores, labels, 0.95)
  assert threshold == 0.6
  assert presence.binary_metrics(scores, labels, threshold)["plantRecall"] == 1
  # Ties/indistinguishable images cannot be disguised as a useful gate.
  threshold = presence.select_threshold([0.5, 0.5], [0, 1], 0.95)
  assert presence.binary_metrics([0.5, 0.5], [0, 1], threshold)["falsePositiveRate"] == 1


def test_single_class_cannot_set_threshold():
  with pytest.raises(ValueError):
    presence.select_threshold([0.9, 0.8], [1, 1], 0.95)


def test_training_respects_pause(dataset):
  root, _ = dataset
  (root / "paused.json").write_text("{}")
  with pytest.raises(ValueError, match="paused"):
    presence.train(root, root / "head.json")


def test_frozen_encoder_replaces_species_head(monkeypatch):
  from types import SimpleNamespace

  import torch

  backbone = torch.nn.Sequential()
  backbone.fc = torch.nn.Linear(2048, 1081)
  backbone.requires_grad_(False).eval()
  pretrained = SimpleNamespace(
    model=backbone,
    manifest={"files": {"resnet50_weights_best_acc.tar": {"sha256": "test"}}},
  )
  monkeypatch.setattr(presence, "PlantNetPretrained", lambda *a, **k: pretrained)
  encoder = presence.PresenceEncoder()
  assert isinstance(encoder.model.fc, torch.nn.Identity)
  assert encoder.identity["preprocessing"] == presence.PREPROCESS_ID


def fake_encoder(monkeypatch):
  import torch

  class Encoder:
    identity = {"dimension": presence.DIMENSION, "testOnly": True}

    def __init__(self, *args, **kwargs):
      pass

    def encode(self, image):
      vector = torch.zeros(presence.DIMENSION)
      vector[0] = 1
      return vector

  monkeypatch.setattr(presence, "PresenceEncoder", Encoder)
  return Encoder


@pytest.mark.parametrize("head_type", ["linear", "mlp32"])
def test_head_roundtrip_no_random_fallback_and_test_not_used_for_training(
  dataset, monkeypatch, head_type,
):
  root, _ = dataset
  # Tiny synthetic caches may live on a tmpfs smaller than the production disk reserve.
  monkeypatch.setattr(presence, "reserve_space", lambda *args: None)
  fake_encoder(monkeypatch)
  original = presence.encode_rows
  encoded_splits = []

  def tracking(encoder, root, rows, **kwargs):
    encoded_splits.extend(row["split"] for row in rows)
    return original(encoder, root, rows, **kwargs)

  monkeypatch.setattr(presence, "encode_rows", tracking)
  head = root / "head.json"
  report = presence.train(root, head, epochs=2, head_type=head_type)
  sidecar = json.loads(head.with_suffix(".metadata.json").read_text())
  assert sidecar["task"] == "plant-presence"
  assert sidecar["plantSet"]["kind"] == "not-applicable"
  assert sidecar["plantSet"]["count"] is None
  assert "test" not in encoded_splits
  assert report["testUsedForSelection"] is False
  assert report["format"] == (2 if head_type == "mlp32" else 1)
  if head_type == "mlp32":
    assert sidecar["classifierArchitecture"] == "2048-32-ReLU-2"
  result = presence.PlantPresenceClassifier(head).predict(Image.new("RGB", (32, 32)))
  assert isinstance(result["containsPlant"], bool)
  assert result["confidence"] is None
  assert result["productionApproved"] is False
  evaluation = presence.evaluate(root, head, root / "report.json")
  assert evaluation["metrics"]["samples"] == 2
  assert evaluation["threshold"] == report["threshold"]
  with pytest.raises(ValueError, match="new"):
    presence.train(root, head, epochs=2)
  with pytest.raises(FileNotFoundError):
    presence.PlantPresenceClassifier(root / "missing-head.json")
  malformed = deepcopy(report)
  malformed["encoder"]["testOnly"] = False
  head.write_text(json.dumps(malformed))
  with pytest.raises(ValueError, match="mismatch"):
    presence.PlantPresenceClassifier(head)


@pytest.mark.parametrize("change", ["nan-weight", "dimension", "classes", "threshold", "nan-bias"])
def test_malformed_head_fails_before_encoder_loading(tmp_path, monkeypatch, change):
  def forbidden(*a, **k):
    raise AssertionError("Malformed head must fail before loading backbone")

  monkeypatch.setattr(presence, "PresenceEncoder", forbidden)
  report = {
    "format": 1, "task": "plant-presence", "labels": presence.LABELS,
    "confidenceCalibrated": False, "threshold": 0.5,
    "weight": [[0.0] * presence.DIMENSION for _ in range(2)], "bias": [0, 0],
  }
  if change == "nan-weight":
    report["weight"][0][0] = float("nan")
  elif change == "dimension":
    report["weight"] = [[0]]
  elif change == "classes":
    report["labels"] = ["plant", "nonplant"]
  elif change == "threshold":
    report["threshold"] = True
  else:
    report["bias"][0] = float("nan")
  path = tmp_path / "head.json"
  path.write_text(json.dumps(report))
  with pytest.raises(ValueError):
    presence.PlantPresenceClassifier(path)


@pytest.mark.parametrize("change", ["architecture", "keys", "dimensions", "nan"])
def test_malformed_mlp_fails_before_encoder_loading(tmp_path, monkeypatch, change):
  import torch

  def forbidden(*a, **k):
    raise AssertionError("Invalid MLP must fail before encoder loading")

  monkeypatch.setattr(presence, "PresenceEncoder", forbidden)
  head = torch.nn.Sequential(
    torch.nn.Linear(presence.DIMENSION, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2),
  )
  report = {
    "format": 2, "task": "plant-presence", "labels": presence.LABELS,
    "confidenceCalibrated": False, "threshold": .5, "headType": "mlp32",
    "state": {name: value.tolist() for name, value in head.state_dict().items()},
  }
  if change == "architecture":
    report["headType"] = "other"
  elif change == "keys":
    report["state"]["extra"] = [0.]
  elif change == "dimensions":
    report["state"]["2.weight"] = [[0.]]
  else:
    report["state"]["0.weight"][0][0] = float("nan")
  path = tmp_path / "head.json"
  path.write_text(json.dumps(report))
  with pytest.raises(ValueError):
    presence.PlantPresenceClassifier(path)


def test_feature_encoding_stops_when_storage_reserve_is_unavailable(tmp_path, monkeypatch):
  from types import SimpleNamespace

  from yizhu_server import snapshot_files

  monkeypatch.setattr(snapshot_files.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))

  class Encoder:
    identity = {"test": True}

    def encode(self, image):
      pytest.fail("Must not encode or write features after exhausting the storage reserve")

  with pytest.raises(OSError, match="4 GiB"):
    presence.cached_feature(Encoder(), tmp_path, {"sha256": "synthetic", "path": "not-read"})
  assert not list(tmp_path.rglob("*.npy"))
