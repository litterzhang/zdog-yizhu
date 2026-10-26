import copy
import json

import pytest

from yizhu_server import partial_species_training as partial

torch = pytest.importorskip("torch")


def test_confirmation_changes_only_training_seed_and_never_mutates_first_recipe():
  original = copy.deepcopy(partial.RECIPE)
  first, second = partial.run_recipe(), partial.run_recipe(True)
  assert first == original
  assert second["seed"] == 20260927
  assert {k for k in first if first[k] != second[k]} == {"seed"}
  second["trainAugmentation"]["horizontalFlip"] = 0
  assert partial.RECIPE == original
  with pytest.raises(ValueError, match="boolean"):
    partial.run_recipe("true")


class TinyPartialModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.early = torch.nn.Sequential(
      torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv2d(4, 8, 1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
    )
    self.fc = torch.nn.Linear(8, 6)

  def forward(self, value):
    return self.fc(self.layer4(self.early(value)).mean((2, 3)))


def test_only_layer4_convolution_and_head_update_with_all_bn_fixed():
  torch.set_num_threads(1)
  torch.manual_seed(8)
  model = TinyPartialModel()
  names = partial.configure_partial_training(model)
  assert set(names) == {"layer4.0.weight", "layer4.0.bias", "fc.weight", "fc.bias"}
  before = {k: v.clone() for k, v in model.state_dict().items()}
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
  for _ in range(2):
    partial.set_partial_mode(model)
    assert not model.early.training
    assert not model.early[1].training
    assert not model.layer4[1].training
    assert model.layer4[0].training and model.fc.training
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(torch.randn(6, 3, 8, 8)), torch.arange(6)).backward()
    optimizer.step()
    model.eval()
  for k, v in model.state_dict().items():
    if k in names:
      assert not torch.equal(v, before[k])
    else:
      assert torch.equal(v, before[k])


@pytest.fixture
def images(tmp_path):
  from PIL import Image

  (tmp_path / "images").mkdir()
  classes = [{"speciesId": f"s{i}", "scientificName": f"Synthetic {i}"} for i in range(6)]
  splits = {"train": [], "validation": []}
  for split in splits:
    for i in range(6):
      path = f"images/{split}-{i}.png"
      Image.new("RGB", (24, 28), (i * 35, 90, 80)).save(tmp_path / path)
      splits[split].append({
        "sampleId": f"{split}-{i}", "speciesId": f"s{i}", "split": split,
        "path": path, "imageSha256": f"fixture-{split}-{i}",
        # Each author spans all classes so cluster resampling covers all species.
        "observerGroup": f"author-{split}",
      })
  return classes, splits


def small_recipe():
  return {**partial.RECIPE, "epochs": 2, "batchSize": 6,
          "pairedBootstrap": {"draws": 20, "seed": 3}}


def test_training_smoke_best_checkpoint_metrics_and_no_test_read(tmp_path, images, monkeypatch):
  from safetensors.torch import load_file

  classes, splits = images
  monkeypatch.setattr(partial, "reserve_space", lambda *a: None)
  torch.manual_seed(9)
  model = TinyPartialModel()
  fitted = partial.fit(model, tmp_path, tmp_path, splits, classes, recipe=small_recipe())
  history = json.loads((tmp_path / "history.json").read_text())
  assert fitted["bestEpoch"] == min(history, key=lambda r: r["validationLoss"])["epoch"]
  assert fitted["epochsRun"] == 2
  assert fitted["optimizerSteps"] == 2
  assert fitted["frozenParametersAndBatchNormUnchanged"] is True
  selected = load_file(tmp_path / "classifier.safetensors")
  assert all(torch.equal(model.state_dict()[k], v) for k, v in selected.items())
  comparison = json.loads((tmp_path / "stage-a-comparison.json").read_text())
  assert comparison["productionApproved"] is False
  assert comparison["automaticConfirmation"] is False
  assert comparison["authorGroups"] == 1
  assert {p["speciesId"] for p in json.loads(
    (tmp_path / "validation-predictions.json").read_text()
  )} == {r["speciesId"] for r in classes}
  assert not (tmp_path / "trained-species.json").exists()  # Wrapper publishes only on completion.


def test_reject_test_rows_before_first_batch(tmp_path, images):
  classes, splits = images
  splits["train"][0]["split"] = "test"
  with pytest.raises(ValueError, match="Test"):
    partial.make_loaders(tmp_path, splits, classes, small_recipe())


def test_baseline_mismatch_stops_before_optimization(tmp_path, images):
  classes, splits = images
  with pytest.raises(ValueError, match="baseline"):
    partial.fit(TinyPartialModel(), tmp_path, tmp_path, splits, classes, recipe=small_recipe(),
                expected_baseline={"macroTop1": -1, "macroTop5": -1, "loss": -1})
  assert not (tmp_path / "classifier.safetensors").exists()


def test_patience_five_keeps_first_checkpoint_on_ties(tmp_path, images, monkeypatch):
  classes, splits = images
  monkeypatch.setattr(partial, "reserve_space", lambda *a: None)
  original = partial.evaluate

  def tied(*args):
    metrics, predictions = original(*args)
    metrics["loss"] = 1.0
    return metrics, predictions

  monkeypatch.setattr(partial, "evaluate", tied)
  result = partial.fit(TinyPartialModel(), tmp_path, tmp_path, splits, classes,
                       recipe={**small_recipe(), "epochs": 15})
  assert result["bestEpoch"] == 1 and result["epochsRun"] == 6


def test_parent_checkpoint_and_class_mapping_are_pinned(tmp_path, monkeypatch):
  classes = [{"speciesId": "s1", "scientificName": "Synthetic a", "chineseName": "示例"}]
  protocol = {"manifestSha256": partial.MANIFEST_SHA, "classes": classes}
  meta = {
    "weightsSha256": partial.PARENT_SHA, "protocolSha256": partial.PARENT_PROTOCOL_SHA,
    "status": "trained-offline", "testUsed": False, "architecture": "resnet18",
    "bestEpoch": 8, "productionApproved": False,
    "classes": [{**classes[0], "trainingExamples": 70}],
  }
  support = {"weightsSha256": partial.PARENT_SHA, "classes": meta["classes"]}
  for name, value in [("protocol", protocol), ("model", meta), ("trained-species", support)]:
    (tmp_path / f"{name}.json").write_text(json.dumps(value))
  monkeypatch.setattr(partial, "sha256", lambda p: (
    partial.PARENT_SHA if p.name == "classifier.safetensors" else partial.PARENT_PROTOCOL_SHA
  ))
  assert partial.validate_parent(tmp_path, classes) == meta
  changed = copy.deepcopy(classes)
  changed[0]["speciesId"] = "not-trained"
  with pytest.raises(ValueError, match="Parent"):
    partial.validate_parent(tmp_path, changed)
  meta["bestEpoch"] = 9
  (tmp_path / "model.json").write_text(json.dumps(meta))
  with pytest.raises(ValueError, match="Parent"):
    partial.validate_parent(tmp_path, classes)
