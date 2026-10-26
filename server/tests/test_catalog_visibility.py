"""Visibility is reference metadata, never deletion of private observations."""

import json
import sqlite3

import pytest

from yizhu_server.app import create_app
from yizhu_server.authclient import AuthError
from yizhu_server.catalog import PlantCatalog, stable_id
from yizhu_server.catalog_import import build_catalog
from yizhu_server.catalog_recognition import ranked_result
from yizhu_server.colchina_names import build_national_index
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.journal_store import JournalStore
from yizhu_server.settings import Settings

HIDDEN = stable_id("tol10m", "Hidden plant")
VISIBLE = "bioclip:plant-001"
ROWS = [
    [["Plantae", "Testophyta", "Testopsida", "Hiddenales", "Hiddenaceae", "Hidden", "plant"], ""],
    [["Plantae", "Testophyta", "Testopsida", "Ginkgoales", "Ginkgoaceae", "Ginkgo", "biloba"], ""],
]


@pytest.fixture
def catalog(tmp_path, national_names):
    path = tmp_path / "catalog"
    build_catalog(ROWS, path, chinese_names=national_names)
    return PlantCatalog(path)


def test_hidden_rows_remain_stored_and_mapped_but_not_public(catalog, tmp_path):
    assert catalog.metadata["visiblePlantCount"] == 1
    assert catalog.metadata["hiddenPlantCount"] == 1
    assert catalog.get(HIDDEN) is None
    assert catalog.get(HIDDEN, include_hidden=True)["isVisible"] is False
    assert catalog.get(VISIBLE)["isVisible"] is True
    assert catalog.plant_indices() == [0, 1]  # Do not bias the model by dropping vectors.
    with sqlite3.connect(catalog.path) as db:
        assert db.execute("SELECT is_visible FROM species WHERE id=?", (HIDDEN,)).fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM species").fetchone()[0] == 2
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    client = create_app(Settings(h5_dist=tmp_path / "public"),
                        recognition_runtime=RecognitionRuntime(store, start=False)).test_client()
    for query in ["", "?chineseOnly=0", "?chineseOnly=1", "?includeHidden=1"]:
        page = client.get("/api/catalog/species" + query).get_json()
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [VISIBLE]
    assert client.get("/api/catalog/species?q=Hidden").get_json()["total"] == 0
    assert client.get("/api/catalog/species/" + HIDDEN).status_code == 404
    nodes = client.get("/api/catalog/taxa?chineseOnly=0").get_json()["items"]
    assert [node["name"] for node in nodes] == ["Ginkgoales"]
    assert nodes[0]["count"] == 1


def test_historical_references_and_unrelated_edits_survive_hiding(catalog, tmp_path):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    original = {"candidates": [{"speciesId": HIDDEN}], "original": {"keep": "raw"}}
    with store.transaction() as db:
        store.put(db, "alice", "plants", {
            "id": "plant", "ownerId": "alice", "speciesId": HIDDEN,
            "nickname": "我的植物", "createdAt": "test", "description": "私人描述",
        })
        store.put(db, "alice", "recognitions", {
            "id": "record", "ownerId": "alice", "createdAt": "test", "status": "succeeded",
            "confirmedSpeciesId": HIDDEN, "rawResult": original,
        })
    snapshot = store.snapshot("alice")
    assert snapshot["species"][0]["isVisible"] is False
    assert snapshot["recognitions"][0]["rawResult"] == original
    edited = store.update_plant("alice", "plant", {"nickname": "新昵称", "speciesId": HIDDEN})
    assert edited["speciesId"] == HIDDEN and edited["description"] == "私人描述"
    with pytest.raises(AuthError):
        store.require_species(HIDDEN)  # Cannot newly select a hidden species.
    with pytest.raises(AuthError):
        store.update_plant("bob", "plant", {"nickname": "越权", "speciesId": HIDDEN})
    assert store.snapshot("bob")["species"] == []


@pytest.mark.parametrize("indices,expected", [([0, 1], "uncertain"), ([1, 0], "succeeded"), ([0], "uncertain")])
def test_hidden_candidates_never_promote_lower_rank_to_success(catalog, tmp_path, indices, expected):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    with store.transaction() as db:
        store.put(db, "alice", "recognitions", {
            "id": "record", "status": "running", "createdAt": "test", "confirmedSpeciesId": None,
        })
    result = ranked_result(catalog, indices, [0.9, 0.8][:len(indices)], top_k=len(indices))
    store.finish("alice", "record", result=result)
    record = store.snapshot("alice")["recognitions"][0]
    assert record["status"] == expected
    assert record["rawResult"]["original"] == result
    assert all(row["speciesId"] != HIDDEN for row in record["rawResult"]["candidates"])
    if indices[0] == 0:
        assert record["rawResult"]["reason"] == "hidden_leading_candidate"
    if len(indices) == 1:
        assert record["rawResult"]["candidates"] == []


def test_adding_chinese_name_restores_visibility_without_changing_identity(catalog, tmp_path, national_names):
    index = build_national_index([{
        "rowNumber": 2, "界拉丁名": "Plantae",
        "物种拉丁名": "Hidden plant", "物种中文名": "示例草",
    }])
    build_catalog(ROWS, tmp_path / "renamed", chinese_names={**national_names, **index})
    new = PlantCatalog(tmp_path / "renamed")
    assert new.get(HIDDEN)["isVisible"] is True
    assert new.get(HIDDEN)["commonName"] == "示例草"
    assert new.metadata["hiddenPlantCount"] == 0
    assert new.labels([0, 1]) == catalog.labels([0, 1])
    assert new.search()["total"] == 2


@pytest.mark.parametrize("name", ["", "  ", "Latin only"])
def test_empty_or_non_chinese_name_does_not_make_row_visible(tmp_path, name):
    build_catalog(ROWS, tmp_path / "catalog", chinese_names={
        ("species", "Hidden plant"): {"name": name, "aliases": [], "source": {}},
    })
    with sqlite3.connect(tmp_path / "catalog") as db:
        row = db.execute("SELECT is_visible,data FROM species WHERE id=?", (HIDDEN,)).fetchone()
    assert row[0] == 0 and json.loads(row[1])["isVisible"] is False
