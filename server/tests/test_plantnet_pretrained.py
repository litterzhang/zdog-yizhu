import hashlib
import json
from types import SimpleNamespace

import pytest

from yizhu_server import plantnet_pretrained as model


def write_maps(root, *, ids=None):
  ids = ids or {"0": "30", "1": "10", "2": "20"}
  (root / "class_idx_to_species_id.json").write_text(json.dumps(ids))
  (root / "plantnet300K_species_id_2_name.json").write_text(json.dumps({
    "10": "Ginkgo biloba L.", "20": "Trifolium repens L.", "30": "Nelumbo nucifera Gaertn.",
  }))


def test_original_output_order_is_not_sorted_by_species_id(tmp_path, monkeypatch):
  monkeypatch.setattr(model, "CLASS_COUNT", 3)
  write_maps(tmp_path)
  assert [identifier for identifier, _ in model.read_class_map(tmp_path)] == ["30", "10", "20"]


@pytest.mark.parametrize("ids", [
  {"0": "30", "1": "10"}, {"0": "30", "2": "10", "3": "20"},
  {"0": "30", "1": "30", "2": "20"}, {"0": "30", "1": "10", "2": "unknown"},
])
def test_incomplete_duplicate_or_incompatible_class_maps_rejected(tmp_path, monkeypatch, ids):
  monkeypatch.setattr(model, "CLASS_COUNT", 3)
  write_maps(tmp_path, ids=ids)
  with pytest.raises(ValueError):
    model.read_class_map(tmp_path)


def test_v2_table_cannot_be_attached_to_original_checkpoint(tmp_path):
  (tmp_path / "class_idx_to_species_id.json").write_text(
    json.dumps({str(i): str(i) for i in range(1000)}),
  )
  (tmp_path / "plantnet300K_species_id_2_name.json").write_text("{}")
  with pytest.raises(ValueError, match="1080"):
    model.read_class_map(tmp_path)


def test_ambiguous_or_infraspecific_names_are_not_silently_species_matches():
  records = model.class_records([
    ("1", "Ginkgo biloba L."), ("2", "Trifolium repens L."),
    ("3", "Trifolium repens Author."), ("4", "Oxalis debilis var. example"),
  ], None)
  assert records[0]["scientificName"] == "Ginkgo biloba"
  assert all(row["scientificName"] is None for row in records[1:])
  assert all(row["catalogSpeciesId"] is None for row in records)


def refs(names):
  return [
    {"sourceSpeciesId": str(i), "scientificName": name,
     "catalogSpeciesId": None, "catalogVisible": False}
    for i, name in enumerate(names)
  ]


def test_unknown_hidden_leader_is_preserved_and_scores_are_not_confidence():
  rows = refs(["Unknown plant", "Ginkgo biloba", "Trifolium repens"])
  result = model.top_candidates(rows, [.8, .15, .05], 2)
  assert result[0]["scientificName"] == "Unknown plant"
  assert all(row["confidence"] is None for row in result)
  assert result[0]["score"] == .8


def test_top5_uses_distinct_species_and_deterministic_tie_order():
  rows = refs(["Ginkgo biloba", "Ginkgo biloba", "Trifolium repens", None])
  result = model.top_candidates(rows, [.3, .3, .3, .1], 3)
  assert [row["sourceSpeciesId"] for row in result] == ["0", "2", "3"]


def test_duplicate_full_source_labels_do_not_occupy_two_top5_slots():
  rows = refs([None, None, "Trifolium repens"])
  rows[0]["sourceScientificName"] = "Example plant Author."
  rows[1]["sourceScientificName"] = "Example plant Author."
  result = model.top_candidates(rows, [.5, .4, .1], 2)
  assert [row["sourceSpeciesId"] for row in result] == ["0", "2"]


@pytest.mark.parametrize("scores", [[1], [float("nan"), 0], [True, 0], [-.1, 1], [2, 0]])
def test_invalid_scores_rejected(scores):
  with pytest.raises(ValueError):
    model.top_candidates(refs(["A a", "B b"]), scores, 2)


def test_missing_classes_and_errors_stay_in_top5_denominator():
  cases = [
    {"expectedScientificName": "A a", "labelCovered": True, "technicalSuccess": True,
     "seconds": 1, "top1Hit": True, "top5Hit": True},
    {"expectedScientificName": "A a", "labelCovered": True, "technicalSuccess": True,
     "seconds": 2, "top1Hit": False, "top5Hit": True},
    {"expectedScientificName": "B b", "labelCovered": False, "technicalSuccess": True,
     "seconds": 3, "top1Hit": False, "top5Hit": False},
    {"expectedScientificName": "C c", "labelCovered": True, "technicalSuccess": False,
     "seconds": 0, "top1Hit": False, "top5Hit": False},
  ]
  report = model.summarize_cases(cases)
  assert report["top1"] == .25 and report["top5"] == .5
  assert report["macroTop5"] == pytest.approx(1 / 3)
  assert report["technicalSuccessRate"] == .75
  assert report["coveredSpecies"] == 2
  assert report["latencyMedianSeconds"] == 2


def test_report_never_overwrites_existing_file(tmp_path):
  path = tmp_path / "result.json"
  model.write_report(path, {"first": True})
  with pytest.raises(FileExistsError):
    model.write_report(path, {"first": False})
  assert json.loads(path.read_text()) == {"first": True}
  assert path.stat().st_mode & 0o777 == 0o600


def test_asset_verification_happens_before_loading_model(tmp_path):
  with pytest.raises(ValueError, match="manifest"):
    model.PlantNetPretrained(tmp_path)


def test_safe_loader_rejects_wrong_head_without_constructing_random_fallback(tmp_path, monkeypatch):
  torch = pytest.importorskip("torch")
  calls = []
  monkeypatch.setattr(model, "verify", lambda *_: {})
  monkeypatch.setattr(model, "read_class_map", lambda *_: [("1", "Ginkgo biloba L.")])

  def load(*args, **kwargs):
    calls.append(kwargs)
    return {"model": {"fc.weight": torch.zeros((1000, 2048))}}

  monkeypatch.setattr(torch, "load", load)
  with pytest.raises(ValueError, match="dimensions"):
    model.PlantNetPretrained(tmp_path)
  assert calls == [{"map_location": "cpu", "weights_only": True}]


def public_row(path="images/1.jpg", *, split="test"):
  return {
    "sampleId": "public-1", "split": split, "path": path, "license": "cc-by",
    "attribution": "Public author", "imageSha256": hashlib.sha256(b"public").hexdigest(),
    "imageUrl": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg",
    "sourceUrl": "https://www.inaturalist.org/observations/1",
  }


def test_evaluation_reuses_only_frozen_public_test_split(tmp_path):
  (tmp_path / "images").mkdir()
  (tmp_path / "images/1.jpg").write_bytes(b"public")
  (tmp_path / "samples").mkdir()
  (tmp_path / "samples/1.json").write_text(json.dumps({"samples": [
    public_row(), {**public_row(split="train"), "sampleId": "training"},
  ]}))
  assert len(model.public_test_rows(tmp_path)) == 1


@pytest.mark.parametrize("change", [
  {"license": "cc-by-nc"}, {"attribution": ""},
  {"sourceUrl": "https://private.example/photo"}, {"imageSha256": "bad"},
  {"path": "../private.jpg"},
])
def test_bad_provenance_or_changed_images_fail_before_evaluation(tmp_path, change):
  (tmp_path / "images").mkdir()
  (tmp_path / "images/1.jpg").write_bytes(b"public")
  (tmp_path / "samples").mkdir()
  (tmp_path / "samples/1.json").write_text(json.dumps({"samples": [{**public_row(), **change}]}))
  with pytest.raises(ValueError):
    model.public_test_rows(tmp_path)


def test_closed_set_controls_are_not_plant_accuracy_or_presence_claims():
  pytest.importorskip("PIL")
  provider = SimpleNamespace(predict=lambda _: [{"scientificName": "Example plant"}])
  result = model.controls(provider)
  assert [row["case"] for row in result["results"]] == ["blank", "noise"]
  assert "not a nonplant accuracy test" in result["purpose"]
