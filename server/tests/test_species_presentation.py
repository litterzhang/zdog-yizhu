"""Display-only tests with synthetic source data and isolated journal/auth fixtures."""

import json
import sqlite3
from copy import deepcopy

import pytest

from yizhu_server.app import create_app
from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.colchina_names import build_national_index
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.journal_store import JournalStore
from yizhu_server.settings import Settings
from yizhu_server.species_presentation import journal_species


def sourced_name(name):
  return {
    "sourceName": name, "recordId": "synthetic:row-1", "url": "https://example.org/source",
    "acceptancePolicy": "visible-sourced-aliases-v1", "matchType": "exact",
  }


def species():
  return {
    "id": "test:plant", "scientificName": "Example species",
    "commonName": "名录名称", "chineseName": "名录名称", "isVisible": True,
    "chineseAliases": ["常用别名"], "nameSources": [sourced_name("常用别名")],
  }


def test_alias_first_does_not_mutate_canonical_or_source_data():
  item = species()
  before = deepcopy(item)
  shown = journal_species(item)
  assert shown["commonName"] == "常用别名"
  assert shown["chineseName"] == "名录名称"
  assert shown["id"] == item["id"]
  assert shown["scientificName"] == item["scientificName"]
  assert shown["nameSources"] == item["nameSources"]
  assert item == before


@pytest.mark.parametrize("overrides,expected", [
  ({"chineseAliases": []}, "名录名称"),
  ({"nameSources": []}, "名录名称"),
  ({"nameSources": [{**sourced_name("常用别名"), "acceptancePolicy": "pending"}]}, "名录名称"),
  ({"nameSources": [{**sourced_name("常用别名"), "matchType": "fuzzy"}]}, "名录名称"),
  ({"chineseName": None, "chineseAliases": []}, "Example species"),
  ({"chineseName": "English name", "chineseAliases": []}, "Example species"),
  ({"isVisible": False}, "名录名称"),
])
def test_fallbacks_and_visibility_are_unchanged(overrides, expected):
  item = {**species(), **overrides}
  shown = journal_species(item)
  assert shown["commonName"] == expected
  assert shown["isVisible"] == item["isVisible"]


def test_multiple_aliases_use_stored_order_not_invented_popularity():
  item = species()
  item["chineseAliases"] = ["未采纳名字", "第二个俗名", "常用别名"]
  item["nameSources"].append(sourced_name("第二个俗名"))
  assert journal_species(item)["commonName"] == "第二个俗名"


def test_existing_recognition_and_confirmed_species_use_alias_without_rewriting_history(tmp_path):
  class Auth:
    configured = True

    def userinfo(self, token):
      return {"sub": token}

  path = tmp_path / "catalog"
  build_catalog(
    [[["Plantae", "Testophyta", "Testopsida", "Testales", "Testaceae", "Example", "species"], ""]],
    path,
    chinese_names=build_national_index([{
      "rowNumber": 2, "界拉丁名": "Plantae", "物种拉丁名": "Example species",
      "物种中文名": "名录名称",
    }]),
  )
  with sqlite3.connect(path) as db:
    item = json.loads(db.execute("SELECT data FROM species").fetchone()[0])
    item.update(chineseAliases=["常用别名"], nameSources=[sourced_name("常用别名")])
    db.execute("UPDATE species SET data=?", (json.dumps(item),))
  catalog = PlantCatalog(path)
  canonical = deepcopy(catalog.get(item["id"]))
  store = JournalStore(tmp_path / "journal", catalog=catalog)
  record = {
    "id": "recognition", "ownerId": "alice", "createdAt": "2026-09-21",
    "status": "succeeded", "confirmedSpeciesId": item["id"],
    "linkedPlantId": "individual", "associationConflict": False,
    "rawResult": {
      "candidates": [{"speciesId": item["id"], "score": 0.4}],
      "original": {"provider": "synthetic", "scientificName": "Example species"},
    },
  }
  plant = {
    "id": "individual", "ownerId": "alice", "createdAt": "2026-09-21",
    "speciesId": item["id"], "nickname": "我家窗边这一株",
  }
  with store.transaction() as db:
    store.put(db, "alice", "recognitions", record)
    store.put(db, "alice", "plants", plant)
  app = create_app(
    Settings(h5_dist=tmp_path / "public"), auth_client=Auth(),
    recognition_runtime=RecognitionRuntime(store, start=False),
  )
  client = app.test_client()
  response = client.get("/api/journal", headers={"Authorization": "Bearer alice"})
  assert response.status_code == 200
  snapshot = response.get_json()
  assert snapshot["species"][0]["commonName"] == "常用别名"
  assert snapshot["species"][0]["chineseName"] == "名录名称"
  assert snapshot["recognitions"] == [record]
  assert snapshot["plants"] == [plant]
  assert client.get("/api/journal").status_code == 401
  assert client.get("/api/journal", headers={"Authorization": "Bearer bob"}).get_json()["species"] == []
  assert client.get(f"/api/catalog/species/{item['id']}").get_json()["commonName"] == "名录名称"
  assert catalog.get(item["id"]) == canonical  # Cached public entries must not be mutated.
  with store.transaction() as db:
    assert store.get(db, "alice", "recognitions", "recognition") == record
    assert store.get(db, "alice", "plants", "individual") == plant
