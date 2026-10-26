"""Locale-separated catalog storage with stable IDs and a legacy read/write view."""

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

SCHEMA_VERSION = 2
CORE_KEYS = ("id", "scientificName", "englishName", "taxonomy", "taxonomyAlternatives")
ZH_KEYS = (
  "commonName", "chineseName", "chineseAliases", "isVisible", "chineseNameOrigin",
  "nameSources", "taxonomyChinese", "knowledge",
)
REMOVE_PATHS = ",".join(f"'$.{key}'" for key in (*CORE_KEYS, *ZH_KEYS))

ENTITIES = """
CREATE TABLE species_core(
  id TEXT PRIMARY KEY,
  scientific_name TEXT NOT NULL,
  english_name TEXT NOT NULL,
  taxonomy_json TEXT NOT NULL CHECK(json_valid(taxonomy_json)),
  taxonomy_alternatives_json TEXT NOT NULL CHECK(json_valid(taxonomy_alternatives_json)),
  growth_form TEXT NOT NULL,
  extra_json TEXT NOT NULL CHECK(json_valid(extra_json))
);
CREATE INDEX species_core_name ON species_core(scientific_name,id);
CREATE INDEX species_core_growth ON species_core(growth_form);
CREATE TABLE species_zh(
  species_id TEXT PRIMARY KEY REFERENCES species_core(id) ON DELETE CASCADE,
  chinese_name TEXT,
  aliases TEXT NOT NULL CHECK(json_valid(aliases) AND json_type(aliases)='array'),
  description TEXT,
  knowledge_metadata TEXT CHECK(knowledge_metadata IS NULL OR json_valid(knowledge_metadata)),
  display_name TEXT,
  name_origin TEXT,
  name_sources TEXT NOT NULL CHECK(json_valid(name_sources)),
  taxonomy_chinese TEXT NOT NULL CHECK(json_valid(taxonomy_chinese)),
  search_text TEXT NOT NULL,
  has_chinese_name INTEGER GENERATED ALWAYS AS
    (CASE WHEN chinese_name GLOB '*[㐀-鿿]*' THEN 1 ELSE 0 END) VIRTUAL,
  is_visible INTEGER NOT NULL CHECK(is_visible IN (0,1) AND is_visible<=has_chinese_name)
);
CREATE INDEX species_zh_visible ON species_zh(is_visible,species_id);
CREATE INDEX species_zh_chinese ON species_zh(chinese_name);
CREATE INDEX species_zh_search ON species_zh(search_text,species_id);
CREATE TABLE taxa_core(id TEXT PRIMARY KEY,rank TEXT,name TEXT,parent TEXT);
CREATE INDEX taxa_core_parent ON taxa_core(parent);
CREATE INDEX taxa_core_name ON taxa_core(rank,name);
CREATE TABLE taxa_zh(
  taxon_id TEXT PRIMARY KEY REFERENCES taxa_core(id) ON DELETE CASCADE,
  chinese_name TEXT,
  name_source TEXT CHECK(name_source IS NULL OR json_valid(name_source))
);
CREATE VIEW taxa AS
SELECT c.id,c.rank,c.name,c.parent,z.chinese_name,z.name_source
FROM taxa_core c LEFT JOIN taxa_zh z ON z.taxon_id=c.id;
CREATE TRIGGER taxa_insert INSTEAD OF INSERT ON taxa BEGIN
  INSERT INTO taxa_core VALUES(NEW.id,NEW.rank,NEW.name,NEW.parent);
  INSERT INTO taxa_zh VALUES(NEW.id,NEW.chinese_name,NEW.name_source);
END;
CREATE TRIGGER taxa_update INSTEAD OF UPDATE ON taxa BEGIN
  SELECT CASE WHEN NEW.id IS NOT OLD.id THEN RAISE(ABORT,'Taxon ID is immutable') END;
  UPDATE taxa_core SET rank=NEW.rank,name=NEW.name,parent=NEW.parent WHERE id=OLD.id;
  INSERT INTO taxa_zh VALUES(OLD.id,NEW.chinese_name,NEW.name_source)
    ON CONFLICT(taxon_id) DO UPDATE SET
      chinese_name=excluded.chinese_name,name_source=excluded.name_source;
END;
CREATE TRIGGER species_core_id_immutable BEFORE UPDATE OF id ON species_core
WHEN NEW.id IS NOT OLD.id BEGIN SELECT RAISE(ABORT,'Species ID is immutable'); END;
CREATE VIEW species AS
SELECT c.id,c.scientific_name,
  COALESCE(z.search_text,c.scientific_name||' '||c.english_name) AS search_text,
  c.growth_form,
  json_set(c.extra_json,
    '$.id',c.id,'$.scientificName',c.scientific_name,'$.englishName',c.english_name,
    '$.taxonomy',json(c.taxonomy_json),
    '$.taxonomyAlternatives',json(c.taxonomy_alternatives_json),
    '$.commonName',COALESCE(z.display_name,z.chinese_name,c.scientific_name),
    '$.chineseName',z.chinese_name,
    '$.chineseAliases',json(COALESCE(z.aliases,'[]')),
    '$.isVisible',json(CASE WHEN z.is_visible=1 THEN 'true' ELSE 'false' END),
    '$.chineseNameOrigin',z.name_origin,
    '$.nameSources',json(COALESCE(z.name_sources,'[]')),
    '$.taxonomyChinese',json(COALESCE(z.taxonomy_chinese,'{}')),
    '$.knowledge',json(CASE WHEN z.knowledge_metadata IS NULL THEN 'null'
      ELSE json_set(z.knowledge_metadata,'$.summary',z.description) END)
  ) AS data,
  COALESCE(z.has_chinese_name,0) AS has_chinese_name,
  COALESCE(z.is_visible,0) AS is_visible
FROM species_core c LEFT JOIN species_zh z ON z.species_id=c.id;
"""


def create_entities(db):
  db.executescript(ENTITIES)
  core_values = f"""
    NEW.id,NEW.scientific_name,COALESCE(json_extract(NEW.data,'$.englishName'),''),
    COALESCE(json_extract(NEW.data,'$.taxonomy'),'null'),
    COALESCE(json_extract(NEW.data,'$.taxonomyAlternatives'),'[]'),
    NEW.growth_form,json_remove(NEW.data,{REMOVE_PATHS})
  """
  zh_values = """
    NEW.id,json_extract(NEW.data,'$.chineseName'),
    COALESCE(json_extract(NEW.data,'$.chineseAliases'),'[]'),
    json_extract(NEW.data,'$.knowledge.summary'),
    CASE WHEN json_type(NEW.data,'$.knowledge')='object'
      THEN json_remove(json_extract(NEW.data,'$.knowledge'),'$.summary') ELSE NULL END,
    json_extract(NEW.data,'$.commonName'),json_extract(NEW.data,'$.chineseNameOrigin'),
    COALESCE(json_extract(NEW.data,'$.nameSources'),'[]'),
    COALESCE(json_extract(NEW.data,'$.taxonomyChinese'),'{}'),
    NEW.search_text,NEW.is_visible
  """
  validate = """
    SELECT CASE WHEN json_extract(NEW.data,'$.id') IS NOT NEW.id
      OR json_extract(NEW.data,'$.scientificName') IS NOT NEW.scientific_name
      THEN RAISE(ABORT,'Catalog identity fields disagree') END;
  """
  columns = ("species_id,chinese_name,aliases,description,knowledge_metadata,display_name,"
             "name_origin,name_sources,taxonomy_chinese,search_text,is_visible")
  db.executescript(f"""
    CREATE TRIGGER species_insert INSTEAD OF INSERT ON species BEGIN
      {validate}
      INSERT INTO species_core VALUES({core_values});
      INSERT INTO species_zh({columns}) VALUES({zh_values});
    END;
    CREATE TRIGGER species_update INSTEAD OF UPDATE ON species BEGIN
      SELECT CASE WHEN NEW.id IS NOT OLD.id THEN RAISE(ABORT,'Species ID is immutable') END;
      {validate}
      INSERT INTO species_core VALUES({core_values})
        ON CONFLICT(id) DO UPDATE SET scientific_name=excluded.scientific_name,
          english_name=excluded.english_name,taxonomy_json=excluded.taxonomy_json,
          taxonomy_alternatives_json=excluded.taxonomy_alternatives_json,
          growth_form=excluded.growth_form,extra_json=excluded.extra_json;
      INSERT INTO species_zh({columns}) VALUES({zh_values})
        ON CONFLICT(species_id) DO UPDATE SET chinese_name=excluded.chinese_name,
          aliases=excluded.aliases,description=excluded.description,
          knowledge_metadata=excluded.knowledge_metadata,display_name=excluded.display_name,
          name_origin=excluded.name_origin,name_sources=excluded.name_sources,
          taxonomy_chinese=excluded.taxonomy_chinese,search_text=excluded.search_text,
          is_visible=excluded.is_visible;
    END;
  """)


def create_catalog_schema(db):
  db.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE metadata(data TEXT NOT NULL);
    CREATE TABLE labels(idx INTEGER PRIMARY KEY,species_id TEXT,kingdom TEXT,name TEXT);
    CREATE INDEX labels_species ON labels(species_id);
    CREATE TABLE membership(species_id TEXT,taxon_id TEXT,PRIMARY KEY(taxon_id,species_id));
    CREATE INDEX membership_species ON membership(species_id);
  """)
  create_entities(db)


def table_digest(db, table, order):
  """Logical comparison, independent of JSON whitespace/key ordering."""
  result = hashlib.sha256()
  count = 0
  cursor = db.execute(f"SELECT * FROM {table} ORDER BY {order}")
  columns = [column[0] for column in cursor.description]
  for values in cursor:
    item = dict(zip(columns, values, strict=True))
    if table in {"species", "_legacy_species"}:
      item["data"] = json.loads(item["data"])
    result.update(json.dumps(item, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode())
    result.update(b"\n")
    count += 1
  return {"rows": count, "sha256": result.hexdigest()}


def migrate(source, output):
  """Copy, split, compare every logical record, then publish a new file; never edit input."""
  if output.exists() or output.is_symlink():
    raise ValueError("Migration output exists; choose a new path.")
  output.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix="catalog-locales-", dir=output.parent)
  os.close(descriptor)
  path = Path(temporary)
  try:
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
      if original.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='species_core'",
      ).fetchone():
        raise ValueError("Catalog is already locale-separated.")
      with closing(sqlite3.connect(path)) as db:
        original.backup(db)
        original_metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
        before = {
          table: table_digest(db, table, order)
          for table, order in (("species", "id"), ("taxa", "id"),
                               ("labels", "idx"), ("membership", "taxon_id,species_id"))
        }
        db.executescript("""
          ALTER TABLE species RENAME TO _legacy_species;
          ALTER TABLE taxa RENAME TO _legacy_taxa;
        """)
        create_entities(db)
        with db:
          db.execute("INSERT INTO species SELECT * FROM _legacy_species")
          db.execute("INSERT INTO taxa SELECT * FROM _legacy_taxa")
        after = {
          table: table_digest(db, table, order)
          for table, order in (("species", "id"), ("taxa", "id"),
                               ("labels", "idx"), ("membership", "taxon_id,species_id"))
        }
        if before != after:
          raise ValueError("Migration changed logical catalog data; refusing to publish.")
        with db:
          db.execute("DROP TABLE _legacy_species")
          db.execute("DROP TABLE _legacy_taxa")
          db.execute("CREATE INDEX IF NOT EXISTS labels_species ON labels(species_id)")
          metadata = dict(original_metadata)
          metadata["storageSchemaVersion"] = SCHEMA_VERSION
          db.execute("UPDATE metadata SET data=?", (json.dumps(metadata, ensure_ascii=False),))
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
          raise ValueError("Migrated catalog failed integrity check.")
        if db.execute("PRAGMA foreign_key_check").fetchall():
          raise ValueError("Migrated catalog has broken foreign keys.")
        db.execute("ANALYZE")
        db.commit()
        db.execute("VACUUM")  # Reclaim the dropped legacy table pages in the staged copy.
      # The input may be the running catalog. Abort if a writer changed logical data.
      for table, order in (("species", "id"), ("taxa", "id"),
                           ("labels", "idx"), ("membership", "taxon_id,species_id")):
        if table_digest(original, table, order) != before[table]:
          raise ValueError("Source changed during migration; retry from a fresh snapshot.")
      if json.loads(original.execute("SELECT data FROM metadata").fetchone()[0]) != original_metadata:
        raise ValueError("Source metadata changed during migration.")
    os.link(path, output)  # Exclusive publish, preserving the original database.
    return {"storageSchemaVersion": SCHEMA_VERSION, "verified": after,
            "sourceUnchanged": True, "integrityCheck": "ok", "foreignKeyErrors": 0}
  finally:
    path.unlink(missing_ok=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(migrate(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
