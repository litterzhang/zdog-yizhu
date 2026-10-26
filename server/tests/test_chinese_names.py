"""Synthetic name records: no network, model, or downloaded dictionary needed."""

import sqlite3

from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.colchina_names import build_national_index
from yizhu_server.journal_store import JournalStore


def test_enriched_catalog_search_hierarchy_scope_and_stable_ids(tmp_path):
    names = [
        [["Plantae", "Tracheophyta", "Magnoliopsida", "Asterales", "Asteraceae", "Helianthus", "annuus"], "sunflower"],
        [["Plantae", "Tracheophyta", "Magnoliopsida", "Asterales", "Asteraceae", "Example", "unknown"], ""],
        [["Plantae", "Tracheophyta", "Magnoliopsida", "Magnoliales", "Magnoliaceae", "Magnolia", "grandiflora"], ""],
    ]
    index = build_national_index([
        {"rowNumber": 2, "界拉丁名": "Plantae", "物种拉丁名": "Helianthus annuus",
         "物种中文名": "向日葵", "目拉丁名": "Asterales", "目中文名": "菊目",
         "科拉丁名": "Asteraceae", "科中文名": "菊科",
         "属拉丁名": "Helianthus", "属中文名": "向日葵属"},
        {"rowNumber": 3, "界拉丁名": "Plantae", "物种拉丁名": "Magnolia grandiflora",
         "物种中文名": "荷花木兰"},
    ])
    build_catalog(names, tmp_path / "before")
    build_catalog(names, tmp_path / "after", chinese_names=index)
    before, after = PlantCatalog(tmp_path / "before"), PlantCatalog(tmp_path / "after")
    assert before.get("bioclip:plant-010") is None
    assert before.get("bioclip:plant-010", include_hidden=True)["commonName"] == "Magnolia grandiflora"
    assert before.get("bioclip:plant-010", include_hidden=True)["chineseNameOrigin"] is None
    assert before.get("bioclip:plant-010", include_hidden=True)["nameSources"] == []
    sunflower = after.search(q="向日葵")["items"][0]
    assert sunflower["commonName"] == "向日葵"
    assert sunflower["scientificName"] == "Helianthus annuus"
    assert sunflower["id"] == before.get(sunflower["id"], include_hidden=True)["id"]
    assert after.search(q="Helianthus annuus")["items"][0]["id"] == sunflower["id"]
    assert sunflower["taxonomyChinese"] == {"order": "菊目", "family": "菊科", "genus": "向日葵属"}
    assert after.get("bioclip:plant-010")["commonName"] == "荷花木兰"
    assert after.get("bioclip:plant-010")["chineseAliases"] == []
    assert after.search(q="广玉兰")["total"] == 0
    assert after.search(q="洋玉兰")["total"] == 0
    assert after.search(chinese_only=True)["total"] == 2
    assert after.search()["total"] == 2
    assert after.metadata["chineseNameCount"] == 2
    order = next(item for item in after.children(chinese_only=True)["items"] if item["name"] == "Asterales")
    assert order["chineseName"] == "菊目" and order["count"] == 1
    assert order["nameSource"]["recordId"] == "植物界-2026-48356!row-2"
    assert after.search(q="unknown")["items"] == []
    with sqlite3.connect(tmp_path / "before") as old, sqlite3.connect(tmp_path / "after") as new:
        assert old.execute("SELECT * FROM labels").fetchall() == new.execute("SELECT * FROM labels").fetchall()
        assert old.execute("SELECT id FROM species ORDER BY id").fetchall() == new.execute("SELECT id FROM species ORDER BY id").fetchall()
    # Renaming reference labels updates history/plant displays, not user content.
    journal = JournalStore(tmp_path / "journal", catalog=before)
    record = {"id": "recognition", "ownerId": "alice", "createdAt": "test",
              "confirmedSpeciesId": sunflower["id"], "linkedPlantId": "plant",
              "associationConflict": False,
              "rawResult": {"candidates": [{"speciesId": sunflower["id"]}],
                            "original": {"scientific_name": "Helianthus annuus"}}}
    plant = {"id": "plant", "ownerId": "alice", "nickname": "门口的那株",
             "speciesId": sunflower["id"], "story": "用户故事", "createdAt": "test"}
    with journal.transaction() as db:
        journal.put(db, "alice", "recognitions", record)
        journal.put(db, "alice", "plants", plant)
    assert journal.snapshot("alice")["species"][0]["commonName"] == "Helianthus annuus"
    journal.catalog = after
    updated = journal.snapshot("alice")
    assert updated["species"][0]["commonName"] == "向日葵"
    assert updated["recognitions"] == [record]
    assert updated["plants"] == [plant]
