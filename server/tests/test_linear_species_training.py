import copy
import json
from pathlib import Path

import pytest

from yizhu_server import initialization_comparison as comparison
from yizhu_server import linear_species_training as linear
from yizhu_server.training_data import sha256

torch = pytest.importorskip("torch")


def test_imagenet_head_is_discarded_and_new_heads_have_identical_initialization(tmp_path):
  from torchvision.models import resnet18

  torch.set_num_threads(1)
  source = resnet18(weights=None)
  with torch.no_grad():
    source.fc.weight.fill_(777)
    source.fc.bias.fill_(777)
  checkpoint = tmp_path / "source.pth"
  torch.save(source.state_dict(), checkpoint)
  a = comparison.build_model("A", count=82)
  b = comparison.build_model("B", count=82, weights=checkpoint, weights_sha=sha256(checkpoint))
  assert torch.equal(a.fc.weight, b.fc.weight)
  assert torch.equal(a.fc.bias, b.fc.bias)
  assert b.fc.out_features == 82
  assert sum(p.numel() for p in b.fc.parameters()) == 42066
  assert not torch.any(b.fc.weight == 777)
  assert torch.equal(source.conv1.weight, b.conv1.weight)
  assert not torch.equal(a.conv1.weight, b.conv1.weight)
  with pytest.raises(ValueError, match="pinned"):
    comparison.build_model("B", count=82, weights=checkpoint, weights_sha="0" * 64)


def test_fixed_validation_preprocessing_is_official():
  import numpy as np
  from PIL import Image
  from torchvision.models import ResNet18_Weights

  image = Image.fromarray(np.random.default_rng(1).integers(
    0, 256, size=(310, 520, 3), dtype=np.uint8,
  ))
  _, validation = comparison.transforms_for_comparison()
  assert torch.equal(validation(image), ResNet18_Weights.IMAGENET1K_V1.transforms()(image))


class TinyBackbone(torch.nn.Module):
  def __init__(self, count):
    super().__init__()
    self.conv = torch.nn.Conv2d(3, 8, 1)
    self.bn = torch.nn.BatchNorm2d(8)
    self.projection = torch.nn.Linear(8, 512)
    self.fc = torch.nn.Linear(512, count)

  def forward(self, value):
    pooled = self.bn(self.conv(value)).mean((2, 3))
    return self.fc(self.projection(pooled))


def image_rows(tmp_path, count=6):
  from PIL import Image

  (tmp_path / "images").mkdir()
  classes = [{"speciesId": f"species:{i}", "scientificName": f"Synthetic {i}"}
             for i in range(count)]
  splits = {"train": [], "validation": []}
  for split in splits:
    for i in range(count):
      for j in range(2):
        path = Path("images") / f"{split}-{i}-{j}.png"
        Image.new("RGB", (24, 30), (i * 30, j * 40, 90)).save(tmp_path / path)
        splits[split].append({
          "sampleId": f"{split}-{i}-{j}", "speciesId": f"species:{i}", "split": split,
          "imageSha256": sha256(tmp_path / path), "path": str(path),
          "observerGroup": f"author-{split}-{i}-{j}",
        })
  return classes, splits


def test_frozen_bn_features_allow_head_backward_and_reject_test(tmp_path):
  from torchvision import transforms as T

  classes, splits = image_rows(tmp_path)
  backbone, head = linear.split_frozen_backbone(TinyBackbone(len(classes)))
  before = comparison.state_digest(backbone.state_dict())
  features, labels = linear.extract_features(
    backbone, tmp_path, splits["train"],
    {r["speciesId"]: i for i, r in enumerate(classes)}, T.ToTensor(), batch_size=4,
  )
  assert features.shape == (12, 512)
  assert not any(p.requires_grad for p in backbone.parameters())
  assert not any(m.training for m in backbone.modules())
  torch.nn.functional.cross_entropy(head(features), labels).backward()
  assert head.weight.grad is not None
  assert comparison.state_digest(backbone.state_dict()) == before
  rows = copy.deepcopy(splits["validation"])
  rows[0]["split"] = "test"
  with pytest.raises(ValueError, match="Test images"):
    linear.extract_features(backbone, tmp_path, rows, {}, T.ToTensor())
  backbone.train()
  with pytest.raises(ValueError, match="evaluation-mode"):
    linear.extract_features(backbone, tmp_path, splits["train"], {}, T.ToTensor())


def test_fit_selects_lowest_validation_loss_and_records_only_actual_classes(tmp_path):
  torch.manual_seed(2)
  classes = [{"speciesId": f"s{i}"} for i in range(6)]
  splits, tensors = {}, {}
  for split in ("train", "validation"):
    labels = torch.arange(6).repeat(4)
    tensors[f"{split}Features"] = torch.nn.functional.one_hot(labels, 512).float()
    tensors[f"{split}Labels"] = labels
    splits[split] = [{
      "sampleId": f"{split}-{i}", "speciesId": f"s{int(label)}", "split": split,
      "observerGroup": f"author-{split}-{i}", "imageSha256": str(i),
    } for i, label in enumerate(labels)]
  head = torch.nn.Linear(512, 6)
  recipe = {**linear.RECIPE, "epochs": 5}
  fitted = linear.fit_head(head, tensors, splits, classes, tmp_path, recipe=recipe)
  history = json.loads((tmp_path / "history.json").read_text())
  validation = json.loads((tmp_path / "validation.json").read_text())
  assert fitted["bestEpoch"] == min(history, key=lambda r: r["validationLoss"])["epoch"]
  assert validation["epoch"] == fitted["bestEpoch"]
  assert fitted["epochsRun"] == 5
  selected, _ = linear.evaluate_head(
    head, tensors["validationFeatures"], tensors["validationLabels"],
    splits["validation"], classes,
  )
  assert selected["loss"] == pytest.approx(validation["loss"])
  predictions = json.loads((tmp_path / "validation-predictions.json").read_text())
  assert all(set(r["top5"]).issubset({c["speciesId"] for c in classes}) for r in predictions)
  splits["train"][0]["split"] = "test"
  with pytest.raises(ValueError, match="Test"):
    linear.fit_head(head, tensors, splits, classes, tmp_path, recipe=recipe)


def test_patience_stops_on_exact_ties_and_keeps_first_checkpoint(tmp_path, monkeypatch):
  # A controlled validation stream tests checkpoint selection, not model quality.
  classes, splits = image_rows(tmp_path)
  tensors = {}
  for split in splits:
    tensors[f"{split}Features"] = torch.randn(len(splits[split]), 512)
    tensors[f"{split}Labels"] = torch.tensor([i // 2 for i in range(12)])
  original = linear.evaluate_head

  def fixed_loss(*args):
    metrics, predictions = original(*args)
    metrics["loss"] = 1.0
    return metrics, predictions

  monkeypatch.setattr(linear, "evaluate_head", fixed_loss)
  result = linear.fit_head(
    torch.nn.Linear(512, 6), tensors, splits, classes, tmp_path,
    recipe={**linear.RECIPE, "epochs": 50, "patience": 8},
  )
  assert result["bestEpoch"] == 1
  assert result["epochsRun"] == 9


def test_full_stage_a_lifecycle_without_real_weights_or_test_images(tmp_path, monkeypatch):
  # Small synthetic smoke test. It is not accuracy evidence for the real 82-class experiment.
  data = tmp_path / "data"
  data.mkdir()
  classes, splits = image_rows(data)
  weights = tmp_path / "fake-source"
  weights.write_text("fixture")
  monkeypatch.setattr(linear, "WEIGHTS_SHA", sha256(weights))
  monkeypatch.setattr(linear, "load_data", lambda data: (classes, splits))
  monkeypatch.setattr(linear, "build_model", lambda *a, **kw: TinyBackbone(len(classes)))
  # This fixture writes kilobytes under /tmp, which may be a small tmpfs.
  # Production's 4 GiB reserve remains enabled on the actual dataset/output filesystem.
  monkeypatch.setattr(linear, "reserve_space", lambda *a: None)
  monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
  monkeypatch.setattr(linear, "RECIPE", {
    **linear.RECIPE, "epochs": 2, "headParameters": 513 * len(classes), "classCount": len(classes),
  })
  output = tmp_path / "experiment"
  linear.train(data, output, weights)
  model = json.loads((output / "model.json").read_text())
  index = json.loads((output / "feature-index.json").read_text())
  support = json.loads((output / "trained-species.json").read_text())
  assert model["status"] == "trained-offline"
  assert model["testUsed"] is False
  assert model["trainBackbone"] is False
  assert model["batchNormStatisticsChanged"] is False
  assert model["productionApproved"] is False
  assert support["classCount"] == len(classes)
  assert all(r["trainingExamples"] == 2 for r in support["classes"])
  assert index["oldFeatureCacheUsed"] is False
  assert index["randomImageAugmentation"] is False
  assert sha256(output / "classifier.safetensors") == model["weightsSha256"]
  assert not (output / "failure.json").exists()
  with pytest.raises(ValueError, match="overwrite"):
    linear.train(data, output, weights)


def test_paired_bootstrap_is_paired_by_author_and_rejects_changed_identity():
  classes = [{"speciesId": f"s{i}"} for i in range(6)]
  a = [{
    "sampleId": f"{author}-{i}", "speciesId": f"s{i}", "observerGroup": f"a{author}",
    "imageSha256": f"{author}-{i}", "top5": [f"s{(i+j) % 6}" for j in range(1, 6)],
  } for author in range(5) for i in range(6)]
  b = copy.deepcopy(a)
  for row in b:
    i = int(row["speciesId"][1:])
    row["top5"] = [f"s{(i+j) % 6}" for j in range(5)]
  result = comparison.paired_comparison(a, b, classes, draws=50)
  assert result["deltaMacroTop5"] == 1.0
  assert result["bootstrap"]["deltaTop5Interval"] == [1.0, 1.0]
  assert result["authorGroups"] == 5
  assert result["productionApproved"] is False
  b[0]["imageSha256"] = "changed"
  with pytest.raises(ValueError, match="identity"):
    comparison.paired_comparison(a, b, classes, draws=50)


@pytest.mark.parametrize("field,value", [
  ("sampleId", "duplicate"), ("split", "test"), ("license", "all-rights-reserved"),
  ("observerGroup", None), ("imageSha256", "0" * 64),
])
def test_frozen_manifest_input_guards(tmp_path, monkeypatch, field, value):
  # The real manifest SHA is pinned. Even a newly pinned fixture must pass row checks.
  from PIL import Image

  image_path = tmp_path / "images"
  image_path.mkdir()
  Image.new("RGB", (2, 2), "green").save(image_path / "a.png")
  classes = [{"speciesId": "a", "scientificName": "Synthetic a"}]
  row = {
    "sampleId": "duplicate", "speciesId": "a", "scientificName": "Synthetic a",
    "split": "train", "license": "cc-by", "attribution": "test author",
    "observerGroup": "author", "observationId": 1, "path": "images/a.png",
    "imageSha256": sha256(image_path / "a.png"),
  }
  row[field] = value
  samples = [row, copy.deepcopy(row)] if field == "sampleId" else [row]
  manifest = {
    "classes": classes, "targetSpeciesIds": ["a"], "testImagesIncluded": False, "samples": samples,
  }
  path = tmp_path / "train-validation.json"
  path.write_text(json.dumps(manifest))
  monkeypatch.setitem(comparison.RECIPE, "classCount", 1)
  with pytest.raises(ValueError):
    comparison.load_data(tmp_path, sha256(path))
