import json
import sqlite3

import pytest

from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.recognition_solutions import SOLUTIONS, get_solution
from yizhu_server.solution_metadata import (
  Exporter,
  MetadataStore,
  artifact_metadata,
  write_artifact_metadata,
)


@pytest.fixture
def catalog(tmp_path, national_names):
  path = tmp_path / "catalog.sqlite3"
  build_catalog([
    [["Plantae", "Test", "Test", "Test", "Test", "Ginkgo", "biloba"], ""],
    [["Plantae", "Test", "Test", "Test", "Test", "Oxalis", "corniculata"], ""],
    [["Plantae", "Test", "Test", "Test", "Test", "Unknown", "species"], ""],
    [["Animalia", "Test", "Test", "Test", "Test", "Animal", "example"], ""],
  ], path, chinese_names=national_names)
  return PlantCatalog(path)


def test_every_registered_solution_has_metadata_and_reranking_does_not_expand_set(tmp_path, catalog):
  directory = tmp_path / "metadata"
  index = Exporter(directory, catalog).export(tmp_path / "empty-server")
  assert set(SOLUTIONS) <= set(index["solutions"])
  store = MetadataStore(directory)
  base = store.get("bioclip-v1")
  assert base["plantSet"]["count"] == 3
  assert base["appPlantSet"]["count"] == 2
  assert store.get("bioclip-v2-rerank")["plantSet"] == base["plantSet"]
  assert store.get("bioclip-v2-whole")["plantSet"] == base["plantSet"]
  assert get_solution("bioclip-v1").metadata(directory) == base
  assert get_solution("bioclip-v1").public_info()["metadataId"] == "bioclip-v1"
  assert base["accuracyGuarantee"] is False


def test_missing_fallback_is_unknown_not_empty_or_false_complete_union(tmp_path, catalog):
  directory = tmp_path / "metadata"
  Exporter(directory, catalog).export(tmp_path / "empty-server")
  meta = MetadataStore(directory).get("bioclip-multiview-plantclef")
  assert meta["plantSet"]["kind"] == "unknown"
  assert meta["plantSet"]["count"] is None
  assert meta["stages"][0]["plantSet"]["count"] == 3
  fallback = meta["stages"][-1]
  assert fallback["plantSet"]["kind"] == "unknown"
  assert fallback["artifact"]["weightsPresent"] is False


def test_missing_catalog_does_not_claim_zero_supported_plants(tmp_path):
  Exporter(tmp_path, PlantCatalog(None)).export(tmp_path / "empty-server")
  store = MetadataStore(tmp_path)
  assert store.get("bioclip-v1")["plantSet"]["count"] is None
  assert store.plants("bioclip-v1")["kind"] == "unknown"
  assert store.plants("bioclip-v1")["total"] is None


def test_snapshot_updates_visibility_but_keeps_old_revisions(tmp_path, catalog):
  root = tmp_path / "snapshots"
  first = Exporter(root, catalog).export(tmp_path / "empty-server")
  before = first["solutions"]["bioclip-v1"]
  with sqlite3.connect(catalog.path) as db:
    db.execute("UPDATE species SET is_visible=0 WHERE scientific_name='Ginkgo biloba'")
  second = Exporter(root, catalog).export(tmp_path / "empty-server")
  after = second["solutions"]["bioclip-v1"]
  assert before["revision"] != after["revision"]
  assert (root / before["file"]).exists()
  assert before["plantCount"] == after["plantCount"] == 3
  assert after["appPlantCount"] == 1
  assert MetadataStore(root).get("bioclip-v1")["appPlantSet"]["count"] == 1


def test_search_chinese_names_pagination_and_hidden_scope(tmp_path, catalog):
  Exporter(tmp_path, catalog).export(tmp_path / "empty-server")
  store = MetadataStore(tmp_path)
  assert store.plants("bioclip-v1", q="银杏")["total"] == 1
  assert store.plants("bioclip-v1", q="GINKGO BILOBA")["total"] == 1
  assert store.plants("bioclip-v1", q="Unknown", scope="model")["total"] == 1
  assert store.plants("bioclip-v1", q="Unknown", scope="app")["total"] == 0
  first, second = store.plants("bioclip-v1", limit=1), store.plants("bioclip-v1", offset=1, limit=1)
  assert first["items"][0]["id"] != second["items"][0]["id"]
  with pytest.raises(ValueError):
    store.plants("bioclip-v1", limit=5000)
  with pytest.raises(ValueError):
    store.get("../secrets")


def test_source_class_order_and_unmapped_names_are_preserved(tmp_path, catalog):
  exporter = Exporter(tmp_path, catalog)
  rows = exporter.mapped_classes("official", [
    ("999", "Ginkgo biloba L."), ("100", "Notincatalog unknown A."),
    ("101", "Oxalis corniculata L."), ("102", "Oxalis corniculata Other."),
  ])
  assert [row["sourceSpeciesId"] for row in rows] == ["999", "100", "101", "102"]
  assert [row["classIndex"] for row in rows] == [0, 1, 2, 3]
  assert rows[0]["catalogSpeciesId"] is not None
  assert rows[1]["catalogSpeciesId"] is None
  assert rows[2]["catalogSpeciesId"] is rows[3]["catalogSpeciesId"] is None
  exporter.add("test", "Test", "species-identification", rows, source={},
               unit="source-output-class")
  assert exporter.index["test"]["plantCount"] == 4  # No forced taxonomic merging.


def test_corrupted_plant_set_is_rejected(tmp_path, catalog):
  Exporter(tmp_path, catalog).export(tmp_path / "empty-server")
  store = MetadataStore(tmp_path)
  path = tmp_path / store.get("bioclip-v1")["plantSet"]["file"]
  path.write_text('{"plants":[]}')
  with pytest.raises(ValueError, match="hash"):
    store.plants("bioclip-v1")


def test_metadata_path_escape_is_rejected(tmp_path, catalog):
  Exporter(tmp_path, catalog).export(tmp_path / "empty-server")
  index = json.loads((tmp_path / "index.json").read_text())
  index["solutions"]["bioclip-v1"]["file"] = "../private.json"
  (tmp_path / "index.json").write_text(json.dumps(index))
  with pytest.raises(ValueError, match="escapes"):
    MetadataStore(tmp_path).get("bioclip-v1")


def test_binary_sidecar_never_claims_all_plants_and_does_not_modify_head(tmp_path):
  head = tmp_path / "head.json"
  head.write_text(json.dumps({
    "task": "plant-presence", "labels": ["nonplant", "plant"],
    "encoder": {"weightsSha256": "synthetic"}, "datasetSha256": "test",
  }))
  before = head.read_bytes()
  meta = write_artifact_metadata(head)
  assert head.read_bytes() == before
  assert meta["plantSet"]["kind"] == "not-applicable"
  assert meta["plantSet"]["plants"] is None and meta["plantSet"]["count"] is None
  assert head.with_suffix(".metadata.json").exists()
  assert write_artifact_metadata(head) == meta
  head.write_text(head.read_text() + " ")
  with pytest.raises(ValueError, match="disagrees"):
    write_artifact_metadata(head)


def test_mlp_binary_metadata_is_discoverable_without_changing_live_gate(tmp_path):
  server = tmp_path / "server"
  head = server / ".cache/models/plant-presence/mlp32-v1/imagenet-resnet50-mlp32.json"
  head.parent.mkdir(parents=True)
  head.write_text(json.dumps({
    "format": 2, "task": "plant-presence", "labels": ["nonplant", "plant"],
    "headType": "mlp32", "encoder": {"weightsSha256": "synthetic"},
    "datasetSha256": "test",
  }))
  directory = tmp_path / "metadata"
  index = Exporter(directory, PlantCatalog(None)).export(server)
  identifier = "plant-presence-mlp32-v1--imagenet-resnet50-mlp32"
  assert identifier in index["solutions"]
  meta = MetadataStore(directory).get(identifier)
  assert meta["source"]["classifierArchitecture"] == "2048-32-ReLU-2"
  assert meta["plantSet"]["kind"] == "not-applicable"
  assert meta["serverSelectable"] is False
  assert meta["source"]["productionApproved"] is False


def test_species_head_metadata_contains_exact_training_classes(tmp_path):
  path = tmp_path / "model.json"
  document = {
    "architecture": "normalized-bioclip2-linear-v1", "encoder": {},
    "weightsSha256": "test", "classCount": 2,
    "classes": [
      {"speciesId": "b", "scientificName": "Test b", "chineseName": "乙"},
      {"speciesId": "a", "scientificName": "Test a", "chineseName": "甲"},
    ],
  }
  path.write_text(json.dumps(document))
  meta = artifact_metadata(path)
  assert [r["id"] for r in meta["plantSet"]["plants"]] == ["b", "a"]
  assert [r["classIndex"] for r in meta["plantSet"]["plants"]] == [0, 1]
  document["classCount"] = 3
  path.write_text(json.dumps(document))
  with pytest.raises(ValueError, match="count"):
    artifact_metadata(path)


def test_declared_pretrained_support_does_not_expand_from_experimental_head(tmp_path, catalog,
                                                                         monkeypatch):
  from yizhu_server import solution_metadata as metadata
  from yizhu_server.training_data import write_json

  server = tmp_path / "server"
  assets = server / ".cache/models/plantnet-300k-pretrained"
  files = {}
  for name in ("class_idx_to_species_id.json", "plantnet300K_species_id_2_name.json"):
    write_json(assets / name, {})
    files[name] = {"sha256": metadata.sha256(assets / name)}
  manifest = {
    "format": 1, "architecture": "resnet50", "classCount": 1081,
    "datasetVariant": "official-1081-class-pretrained-not-V2", "files": files,
  }
  write_json(assets / "resnet50.manifest.json", manifest)
  monkeypatch.setattr(metadata, "read_class_map", lambda _: [
    ("1", "Ginkgo biloba"), ("2", "Oxalis corniculata"),
  ])
  with catalog.connect() as db:
    identifier = db.execute(
      "SELECT id FROM species WHERE scientific_name='Ginkgo biloba'",
    ).fetchone()["id"]
  directory = tmp_path / "metadata"
  member = {
    "speciesId": identifier, "scientificName": "Ginkgo biloba", "chineseName": "银杏",
    "sourceClasses": [{
      "classIndex": 0, "sourceSpeciesId": "1", "sourceScientificName": "Ginkgo biloba",
    }],
  }
  contract = {
    "schemaVersion": 1, "solutionId": "plantnet-300k-resnet50", "status": "pretrained",
    "sourceFiles": files, "supportedSpecies": [member],
  }
  path = directory / "contracts/plantnet-300k-resnet50.json"
  write_json(path, contract)
  first = Exporter(directory, catalog).export(server)["solutions"]["plantnet-300k-resnet50"]
  assert first["supportedPlantCount"] == 1
  store = MetadataStore(directory)
  assert store.plants("plantnet-300k-resnet50", scope="model")["total"] == 2
  assert store.plants("plantnet-300k-resnet50", scope="supported")["total"] == 1
  assert store.plants("plantnet-300k-resnet50", scope="supported", q="Oxalis")["total"] == 0
  # Discovery of an unrelated trained experimental head cannot mutate this solution's support.
  write_json(server / ".artifacts/models/experiment/model.json", {
    "architecture": "normalized-bioclip2-linear-v1", "encoder": {},
    "weightsSha256": "synthetic", "classCount": 1,
    "classes": [{"speciesId": "another-id", "scientificName": "Test species"}],
  })
  second = Exporter(directory, catalog).export(server)["solutions"]["plantnet-300k-resnet50"]
  assert second == first
  contract["supportedSpecies"].append({
    "speciesId": "not-trained", "scientificName": "Test species", "chineseName": "未训练",
    "sourceClasses": [{"classIndex": 5, "sourceSpeciesId": "x", "sourceScientificName": "Test species"}],
  })
  write_json(path, contract)
  with pytest.raises(ValueError, match="trained output"):
    Exporter(directory, catalog).export(server)
