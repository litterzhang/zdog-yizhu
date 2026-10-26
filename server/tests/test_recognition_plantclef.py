import argparse
import hashlib
import io
import json
import tarfile

import pytest

from yizhu_server import recognition_plantclef as plantclef


def test_scientific_name_mapping_is_conservative_and_never_uses_chinese_aliases(monkeypatch):
  monkeypatch.setattr(plantclef, "CLASS_COUNT", 4)
  names = {
    "1": "Example plant (L.) Test.", "2": "Example plant subsp. minor Test.",
    "3": "Other × hybrid Test.", "4": "Missing plant L.",
  }
  rows = plantclef.class_references(["1", "2", "3", "4"], names, [("stable-id", "Example plant")])
  assert rows[0]["species_id"] == "stable-id"
  assert rows[0]["source_species_id"] == "1"
  assert [row["species_id"] for row in rows[1:]] == [None, None, None]
  assert rows[0]["class_index"] == 0


def test_duplicate_classes_or_taxon_names_are_not_silently_merged(monkeypatch):
  monkeypatch.setattr(plantclef, "CLASS_COUNT", 2)
  with pytest.raises(ValueError):
    plantclef.class_references(["1", "1"], {"1": "Example plant L."}, [])
  names = {"1": "Example plant L.", "2": "Example plant Other"}
  assert all(row["species_id"] is None for row in plantclef.class_references(
    ["1", "2"], names, [("id", "Example plant")],
  ))
  names["2"] = "Other plant L."
  assert plantclef.class_references(
    ["1", "2"], names, [("id-one", "Example plant"), ("id-two", "Example plant")],
  )[0]["species_id"] is None


def test_classifier_keeps_unmapped_leader_and_never_reports_softmax_as_confidence():
  refs = [
    {"species_id": None, "scientific_name": "Unmapped plant", "class_index": 0},
    {"species_id": "id", "scientific_name": "Example plant", "class_index": 1},
  ]
  manifest = {"files": {"model.safetensors": {"sha256": "test"}}}
  result = plantclef.class_result(refs, [0.99, 0.01], limit=2, manifest=manifest)
  assert result["candidates"][0]["kind"] == "unmapped"
  assert result["candidates"][1]["species_id"] == "id"
  assert all(row["confidence"] is None for row in result["candidates"])
  with pytest.raises(ValueError):
    plantclef.class_result(refs, [float("nan"), 1], limit=2, manifest=manifest)


def test_missing_assets_fail_before_model_import_or_network(tmp_path):
  with pytest.raises(ValueError, match="Prepare"):
    plantclef.PlantCLEFProvider(None, directory=tmp_path)


def test_preparer_refuses_unknown_archive_and_existing_destination(tmp_path):
  source, output = tmp_path / "source.tar", tmp_path / "out"
  source.write_bytes(b"not the publisher archive")
  with pytest.raises(ValueError, match="checksum"):
    plantclef.prepare_archive(source, output)
  assert not output.exists()
  output.mkdir()
  with pytest.raises(ValueError, match="exists"):
    plantclef.prepare_archive(source, output)


def test_safe_preparation_extracts_only_allowed_files_and_saves_ema(tmp_path, monkeypatch):
  torch = pytest.importorskip("torch")
  pytest.importorskip("safetensors")
  monkeypatch.setattr(plantclef, "CLASS_COUNT", 2)
  checkpoint = io.BytesIO()
  torch.save({
    "state_dict_ema": {"head.weight": torch.ones(2, 768), "head.bias": torch.zeros(2)},
    "state_dict": {"head.weight": torch.zeros(2, 768), "head.bias": torch.ones(2)},
    "args": argparse.Namespace(model="ignored-publisher-setting", img_size=999),
  }, checkpoint)
  assets = {
    "class_mapping.txt": b"1\n2\n",
    "species_id_to_name.txt": b'"species_id";"species"\n"1";"Example plant L."\n"2";"Other plant L."\n',
    "checkpoint.pth.tar": checkpoint.getvalue(),
  }
  members = {
    name: ("published/" + name, len(raw), hashlib.sha256(raw).hexdigest() if name != "checkpoint.pth.tar" else None)
    for name, raw in assets.items()
  }
  archive = tmp_path / "official-fixture.tar"
  with tarfile.open(archive, "w") as tar:
    for name, raw in [*assets.items(), ("../outside.txt", b"not extracted")]:
      info = tarfile.TarInfo("published/" + name)
      info.size = len(raw)
      tar.addfile(info, io.BytesIO(raw))
  monkeypatch.setattr(plantclef, "ARCHIVE_BYTES", archive.stat().st_size)
  monkeypatch.setattr(plantclef, "ARCHIVE_MD5", plantclef.checksum(archive, "md5"))
  monkeypatch.setattr(plantclef, "MEMBERS", members)
  output = tmp_path / "prepared"
  manifest = plantclef.prepare_archive(archive, output)
  assert manifest["weights"] == "state_dict_ema"
  assert not (tmp_path / "outside.txt").exists()
  assert not (output / "checkpoint.pth.tar").exists()
  assert plantclef.verify_assets(output) == manifest
  from safetensors.torch import load_file

  assert torch.equal(load_file(output / "model.safetensors")["head.weight"], torch.ones(2, 768))
  with (output / "model.safetensors").open("ab") as f:
    f.write(b"corruption")
  with pytest.raises(ValueError, match="checksum"):
    plantclef.verify_assets(output)


def test_wrong_manifest_revision_is_rejected(tmp_path):
  (tmp_path / "manifest.json").write_text(json.dumps({"format": 1, "record": "wrong"}))
  with pytest.raises(ValueError, match="manifest"):
    plantclef.verify_assets(tmp_path)
