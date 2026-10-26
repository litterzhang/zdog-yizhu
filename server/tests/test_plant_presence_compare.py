import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from yizhu_server import plant_presence as presence
from yizhu_server import plant_presence_compare as comparison
from yizhu_server import plant_presence_data as data
from yizhu_server.plant_presence_compare import choose_on_validation
from yizhu_server.training_data import group_split


def test_selection_uses_validation_not_test_results():
  heads = {
    "plantnet": {"validation": {"plantRecall": 0.96, "falsePositiveRate": 0.3},
                "test": {"accuracy": 1}},
    "imagenet": {"validation": {"plantRecall": 0.95, "falsePositiveRate": 0.2},
                "test": {"accuracy": 0}},
  }
  assert choose_on_validation(heads) == "imagenet"
  heads["imagenet"]["validation"]["plantRecall"] = 0.94
  assert choose_on_validation(heads) == "plantnet"
  heads["plantnet"]["validation"]["plantRecall"] = 0.94
  with pytest.raises(ValueError):
    choose_on_validation(heads)


def test_imagenet_pin_safe_loading_and_strict_state(tmp_path, monkeypatch):
  import torch
  from torchvision import models

  monkeypatch.setattr(presence, "DIMENSION", 2)
  model = torch.nn.Module()
  model.fc = torch.nn.Linear(2, 1000)
  path = tmp_path / presence.IMAGENET_FILENAME
  torch.save(model.state_dict(), path)
  monkeypatch.setattr(presence, "IMAGENET_BYTES", path.stat().st_size)
  monkeypatch.setattr(presence, "IMAGENET_SHA256", presence.sha256(path))
  original = torch.load
  calls = []

  def load(*args, **kwargs):
    calls.append(kwargs)
    return original(*args, **kwargs)

  def make(*, weights, num_classes):
    assert weights is None and num_classes == 1000
    return model

  monkeypatch.setattr(torch, "load", load)
  monkeypatch.setattr(models, "resnet50", make)
  result = presence.imagenet_model(tmp_path)
  assert calls == [{"map_location": "cpu", "weights_only": True}]
  assert not result.training and not any(p.requires_grad for p in result.parameters())
  path.write_bytes(b"corrupt")
  with pytest.raises(ValueError, match="missing or changed"):
    presence.imagenet_model(tmp_path)
  assert len(calls) == 1


def test_feature_cache_separates_backbone_and_image_identity(tmp_path, monkeypatch):
  import torch
  from PIL import Image

  monkeypatch.setattr(presence, "reserve_space", lambda *args: None)
  (tmp_path / "images").mkdir()
  photo = tmp_path / "images/a.png"
  Image.new("RGB", (12, 12), "green").save(photo)
  row = {"path": "images/a.png", "sha256": presence.sha256(photo)}

  class Encoder:
    def __init__(self, name):
      self.identity = {"testBackbone": name}
      self.calls = 0

    def encode(self, image):
      self.calls += 1
      vector = torch.zeros(presence.DIMENSION)
      vector[0 if self.identity["testBackbone"] == "a" else 1] = 1
      return vector

  a, b = Encoder("a"), Encoder("b")
  assert presence.cached_feature(a, tmp_path, row)[0] == 1
  assert presence.cached_feature(a, tmp_path, row)[0] == 1
  assert presence.cached_feature(b, tmp_path, row)[1] == 1
  assert a.calls == b.calls == 1
  assert len(list((tmp_path / "features").glob("*/*.npy"))) == 2


def test_previous_test_authors_and_original_duplicates_are_excluded(tmp_path):
  meta = tmp_path / "metadata"
  meta.mkdir()
  (meta / data.CLASS_FILE).write_text(f"{data.PLANT},Plant\n")
  (meta / data.HIERARCHY_FILE).write_text(json.dumps({"LabelName": data.PLANT}))
  images, annotations = [], []
  for index in range(120):
    image_id = f"{index:016x}"
    images.append({
      "ImageID": image_id, "License": data.LICENSE,
      "AuthorProfileURL": f"https://example.org/author/{index}",
      "Author": "Synthetic author", "OriginalMD5": f"md5-{index}",
      "OriginalLandingURL": f"https://example.org/image/{index}",
    })
    annotations.append({
      "ImageID": image_id, "LabelName": data.PLANT,
      "Confidence": str(index % 2), "Source": "verification",
    })
  protected = images[0]
  # A new ID and author must not reintroduce the old test image's original MD5.
  images[1]["OriginalMD5"] = protected["OriginalMD5"]
  for name, rows in ((data.IMAGE_FILE, images), (data.LABEL_FILE, annotations)):
    with (meta / name).open("w") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  previous = tmp_path / "previous"
  previous.mkdir()
  group = hashlib.sha256(
    f"openimages-author:{protected['AuthorProfileURL']}".encode(),
  ).hexdigest()
  (previous / "samples.json").write_text(json.dumps({"samples": [{
    "id": f"openimages:validation:{protected['ImageID']}", "split": "test", "group": group,
  }]}))
  selection = data.select(tmp_path, {split: 2 for split in data.TARGETS}, previous=previous)
  assert len(selection["samples"]) == 12
  assert all(row["group"] != group for row in selection["samples"])
  assert not {r["id"].rsplit(":", 1)[-1] for r in selection["samples"]} & {
    images[0]["ImageID"], images[1]["ImageID"],
  }
  assert all(row["split"] == group_split(row["group"]) for row in selection["samples"])
  assert set(Counter((r["split"], r["label"]) for r in selection["samples"]).values()) == {2}
  assert Path(selection["previousDataset"]["path"]) == previous.resolve()


def test_pipeline_stops_immediately_when_download_fails(tmp_path, monkeypatch):
  root, reports = tmp_path / "data", tmp_path / "reports"
  root.mkdir()
  (root / "selection.json").write_text("{}")
  calls = []

  def fail(*args, **kwargs):
    calls.append("download")
    raise TimeoutError("Synthetic network timeout")

  def forbidden(*args, **kwargs):
    raise AssertionError("Must not start training after failed data preparation")

  monkeypatch.setattr(data, "collect", fail)
  monkeypatch.setattr(comparison, "run", forbidden)
  with pytest.raises(TimeoutError):
    comparison.run_pipeline(root, tmp_path / "models", reports, prepare_data=True)
  status = json.loads((reports / "run-status.json").read_text())
  assert status["stage"] == "failed" and status["errorType"] == "TimeoutError"
  assert calls == ["download"]


def test_pipeline_download_then_training_without_polling(tmp_path, monkeypatch):
  root, reports = tmp_path / "data", tmp_path / "reports"
  root.mkdir()
  (root / "selection.json").write_text("{}")
  calls = []

  def collect(*args, **kwargs):
    calls.append("download")
    kwargs["progress"]({"processedImages": 6, "totalImages": 6})
    (root / "samples.json").write_text("{}")

  def run(*args, **kwargs):
    assert (root / "samples.json").exists()
    calls.append("train")
    kwargs["progress"]({"stage": "training", "backbone": "synthetic-test-only"})
    return {"chosenBackbone": "synthetic-test-only", "testMetrics": {}}

  monkeypatch.setattr(data, "collect", collect)
  monkeypatch.setattr(comparison, "run", run)
  comparison.run_pipeline(root, tmp_path / "models", reports, prepare_data=True)
  assert calls == ["download", "train"]
  assert json.loads((reports / "run-status.json").read_text())["stage"] == "complete"


def test_pipeline_never_recollects_a_frozen_dataset(tmp_path, monkeypatch):
  (tmp_path / "samples.json").write_text("{}")

  def forbidden(*args, **kwargs):
    raise AssertionError("Frozen dataset must not be redownloaded")

  monkeypatch.setattr(data, "collect", forbidden)
  monkeypatch.setattr(comparison, "run", lambda *a, **k: {
    "chosenBackbone": "synthetic-test-only", "testMetrics": {},
  })
  comparison.run_pipeline(
    tmp_path, tmp_path / "models", tmp_path / "reports", prepare_data=True,
  )
