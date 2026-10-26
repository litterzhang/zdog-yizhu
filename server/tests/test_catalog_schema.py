import json
import sqlite3

import pytest

from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.catalog_schema import CORE_KEYS, ZH_KEYS, migrate, table_digest


@pytest.fixture
def legacy_catalog(tmp_path, national_names):
  modern = tmp_path / "new.sqlite3"
  build_catalog([
    [["Plantae", "Testophyta", "Testopsida", "Ginkgoales", "Ginkgoaceae", "Ginkgo", "biloba"], "Ginkgo"],
    [["Plantae", "Testophyta", "Testopsida", "Oxalidales", "Oxalidaceae", "Oxalis", "corniculata"], ""],
    [["Plantae", "Testophyta", "Testopsida", "Otherales", "Otheraceae", "Unknown", "plant"], ""],
  ], modern, chinese_names=national_names, knowledge={
    "Ginkgo biloba": {
      "summary": "合成中文简介", "isAiGenerated": False, "reviewStatus": "unreviewed",
      "sources": [{"title": "Synthetic source", "url": "https://example.org/source"}],
      "extraNullable": None,
    },
  })
  legacy = tmp_path / "legacy.sqlite3"
  with sqlite3.connect(modern) as source, sqlite3.connect(legacy) as target:
    target.executescript("""
      CREATE TABLE metadata(data TEXT);
      CREATE TABLE species(id TEXT PRIMARY KEY,scientific_name TEXT,search_text TEXT,
        growth_form TEXT,data TEXT,has_chinese_name INTEGER,is_visible INTEGER);
      CREATE TABLE taxa(id TEXT PRIMARY KEY,rank TEXT,name TEXT,parent TEXT,
        chinese_name TEXT,name_source TEXT);
      CREATE TABLE labels(idx INTEGER PRIMARY KEY,species_id TEXT,kingdom TEXT,name TEXT);
      CREATE TABLE membership(species_id TEXT,taxon_id TEXT,PRIMARY KEY(taxon_id,species_id));
    """)
    for table in ("metadata", "species", "taxa", "labels", "membership"):
      cursor = source.execute(f"SELECT * FROM {table}")
      marks = ",".join("?" for _ in cursor.description)
      target.executemany(f"INSERT INTO {table} VALUES({marks})", cursor.fetchall())
    data = json.loads(target.execute(
      "SELECT data FROM species WHERE id='bioclip:plant-001'",
    ).fetchone()[0])
    data["chineseAliases"] = ["合成俗名"]
    data["independentReference"] = {"quality": None, "verified": False}
    target.execute("UPDATE species SET data=? WHERE id='bioclip:plant-001'", (
      json.dumps(data, ensure_ascii=False),
    ))
  return legacy


def test_migration_is_lossless_and_never_edits_source(legacy_catalog, tmp_path):
  original = legacy_catalog.read_bytes()
  output = tmp_path / "migrated.sqlite3"
  report = migrate(legacy_catalog, output)
  assert legacy_catalog.read_bytes() == original
  assert report["sourceUnchanged"] and report["foreignKeyErrors"] == 0
  assert report["verified"]["species"]["rows"] == 3
  with sqlite3.connect(output) as new, sqlite3.connect(legacy_catalog) as old:
    for table, order in (("species", "id"), ("taxa", "id"), ("labels", "idx"),
                         ("membership", "taxon_id,species_id")):
      assert table_digest(new, table, order) == table_digest(old, table, order)
    assert new.execute(
      "SELECT type FROM sqlite_master WHERE name='species'",
    ).fetchone()[0] == "view"
    assert new.execute("SELECT count(*) FROM species_core").fetchone()[0] == 3
    assert new.execute("SELECT count(*) FROM species_zh").fetchone()[0] == 3
    core = json.loads(new.execute(
      "SELECT extra_json FROM species_core WHERE id='bioclip:plant-001'",
    ).fetchone()[0])
    assert not set(core) & set((*CORE_KEYS, *ZH_KEYS))
    assert core["independentReference"] == {"quality": None, "verified": False}
    assert new.execute(
      "SELECT chinese_name,description FROM species_zh WHERE species_id='bioclip:plant-001'",
    ).fetchone() == ("银杏", "合成中文简介")
  assert PlantCatalog(output).search(q="银杏")["total"] == 1


def test_chinese_edits_do_not_change_base_identity_or_taxonomy(legacy_catalog, tmp_path):
  output = tmp_path / "migrated.sqlite3"
  migrate(legacy_catalog, output)
  with sqlite3.connect(output) as db:
    before = db.execute("SELECT * FROM species_core ORDER BY id").fetchall()
    db.execute("""
      UPDATE species_zh SET chinese_name='新中文名',aliases='["新俗名"]',
        description='新的有来源描述',display_name='新中文名',search_text='Ginkgo biloba 新中文名 新俗名'
      WHERE species_id='bioclip:plant-001'
    """)
    assert db.execute("SELECT * FROM species_core ORDER BY id").fetchall() == before
  item = PlantCatalog(output).get("bioclip:plant-001")
  assert item["scientificName"] == "Ginkgo biloba"
  assert item["chineseAliases"] == ["新俗名"]
  assert item["knowledge"]["summary"] == "新的有来源描述"
  assert item["knowledge"]["sources"][0]["title"] == "Synthetic source"
  assert PlantCatalog(output).search(q="新俗名")["total"] == 1


def test_taxonomy_translations_join_by_stable_node_ids(legacy_catalog, tmp_path):
  output = tmp_path / "migrated.sqlite3"
  migrate(legacy_catalog, output)
  with sqlite3.connect(output) as db:
    row = db.execute("""
      SELECT c.id,c.name FROM membership m JOIN taxa_core c ON c.id=m.taxon_id
      WHERE m.species_id='bioclip:plant-001' AND c.rank='family'
    """).fetchone()
    db.execute("UPDATE taxa_zh SET chinese_name='合成银杏科' WHERE taxon_id=?", (row[0],))
    assert db.execute("SELECT name,chinese_name FROM taxa WHERE id=?", (row[0],)).fetchone() == (
      "Ginkgoaceae", "合成银杏科",
    )


def test_stable_id_and_localization_foreign_key_are_enforced(legacy_catalog, tmp_path):
  output = tmp_path / "migrated.sqlite3"
  migrate(legacy_catalog, output)
  with sqlite3.connect(output) as db:
    db.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
      db.execute("UPDATE species_core SET id='changed' WHERE id='bioclip:plant-001'")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
      db.execute("""
        INSERT INTO species_zh(species_id,chinese_name,aliases,name_sources,
          taxonomy_chinese,search_text,is_visible)
        VALUES('absent','草','[]','[]','{}','草',1)
      """)
    with pytest.raises(sqlite3.IntegrityError):
      db.execute("UPDATE species_zh SET chinese_name=NULL WHERE is_visible=1")


def test_repeat_migration_and_existing_output_are_rejected(legacy_catalog, tmp_path):
  output = tmp_path / "migrated.sqlite3"
  migrate(legacy_catalog, output)
  with pytest.raises(ValueError, match="exists"):
    migrate(legacy_catalog, output)
  with pytest.raises(ValueError, match="already"):
    migrate(output, tmp_path / "again.sqlite3")
  assert not (tmp_path / "again.sqlite3").exists()


def test_inconsistent_legacy_visibility_does_not_publish(legacy_catalog, tmp_path):
  with sqlite3.connect(legacy_catalog) as db:
    db.execute("UPDATE species SET is_visible=0 WHERE id='bioclip:plant-001'")
  output = tmp_path / "bad.sqlite3"
  with pytest.raises(ValueError, match="changed logical"):
    migrate(legacy_catalog, output)
  assert not output.exists()
