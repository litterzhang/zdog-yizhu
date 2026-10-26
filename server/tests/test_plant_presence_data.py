import json

import pytest

from yizhu_server import plant_presence_data
from yizhu_server.plant_presence_data import PLANT, explicit_labels, walk_hierarchy


def record(image, label=PLANT, confidence="1.0", source="verification"):
  return {"ImageID": image, "LabelName": label, "Confidence": confidence, "Source": source}


def test_explicit_human_positive_and_negative_are_kept():
  result = explicit_labels([
    record("yes"), record("no", confidence="0.0"),
    record("crowd", source="crowdsource-verification", confidence="1"),
  ], {PLANT})
  assert result["yes"]["label"] == result["crowd"]["label"] == "plant"
  assert result["no"]["label"] == "nonplant"
  assert result["no"]["confidence"] == 0


def test_missing_plant_label_never_becomes_negative():
  result = explicit_labels([record("animal-photo", label="bird")], {PLANT})
  assert result == {}


def test_machine_and_fractional_labels_not_ground_truth():
  assert explicit_labels([
    record("machine", source="machine", confidence="1"),
    record("machine-no", source="machine", confidence="0"),
    record("fraction", confidence="0.9"),
  ], {PLANT}) == {}


def test_contradictory_plant_labels_excluded():
  assert explicit_labels([
    record("conflict"), record("conflict", confidence="0"),
  ], {PLANT}) == {}


def test_negative_plant_with_positive_tree_or_grass_excluded():
  result = explicit_labels([
    record("tree", confidence="0"), record("tree", label="tree", confidence="1"),
    record("grass", confidence="0"), record("grass", label="grass", confidence="1"),
    record("real-negative", confidence="0"), record("real-negative", label="bird"),
  ], {PLANT, "tree", "grass"})
  assert set(result) == {"real-negative"}


def test_duplicate_source_votes_not_duplicate_samples():
  result = explicit_labels([
    record("same"), record("same", source="crowdsource-verification"),
  ], {PLANT})
  assert len(result) == 1
  assert result["same"]["sources"] == ["crowdsource-verification", "verification"]


def test_hierarchy_includes_nested_descendants_and_parts():
  tree = {"LabelName": PLANT, "Subcategory": [
    {"LabelName": "tree", "Subcategory": [{"LabelName": "maple"}],
     "Part": [{"LabelName": "leaf"}]},
  ]}
  assert {node["LabelName"] for node in walk_hierarchy(tree)} == {PLANT, "tree", "maple", "leaf"}


def test_unsafe_image_is_explicitly_excluded(monkeypatch, tmp_path):
  def reject(*args):
    raise ValueError("Photo exceeds the 20-megapixel limit.")

  monkeypatch.setattr(plant_presence_data, "download_one", reject)
  result = plant_presence_data.download_sample(tmp_path, {"id": "test-id"})
  assert result["id"] == "test-id"
  assert "20-megapixel" in result["excludedReason"]


def test_access_denial_stops_download_instead_of_becoming_dataset_exclusion(monkeypatch, tmp_path):
  def reject(*args):
    raise PermissionError("Denied")

  monkeypatch.setattr(plant_presence_data, "download_one", reject)
  with pytest.raises(PermissionError):
    plant_presence_data.download_sample(tmp_path, {"id": "test-id"})


def test_collection_does_not_keep_requesting_after_access_denial(monkeypatch, tmp_path):
  (tmp_path / "selection.json").write_text(json.dumps({
    "samples": [{"id": str(index)} for index in range(20)],
  }))
  calls = []

  def reject(root, row):
    calls.append(row["id"])
    raise PermissionError("Rate limited")

  monkeypatch.setattr(plant_presence_data, "download_sample", reject)
  with pytest.raises(PermissionError):
    plant_presence_data.collect(tmp_path, workers=1)
  assert calls == ["0"]
  assert not (tmp_path / "samples.json").exists()
