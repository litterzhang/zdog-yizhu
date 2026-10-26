"""Apply sourced aliases to a new catalog snapshot, keeping primary names and IDs."""

import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path

from yizhu_server.name_enrichment import BINOMIAL, HAN, SCIDB_SOURCE, digest, scidb_rows

POLICY = "visible-sourced-aliases-v1"
PRIMARY_POLICY = "missing-primary-sciencedb-exact-v1"


def alias_source(candidate: dict, record: dict, source: dict, item: dict) -> dict | None:
  """Accept known provenance only; never promote a source primary name implicitly."""
  flags = set(json.loads(candidate["flags"]))
  if flags - {"rank_inferred_from_binomial"}:
    return None
  if (candidate["match_status"] != "exact" or record["rank"] != "species"
      or record["scientificName"] != item["scientificName"]
      or not item.get("isVisible") or item.get("chineseNameOrigin") != "colchina"):
    return None
  name = candidate["name"]
  if not any(n["name"].strip() == name and n["kind"] == candidate["kind"]
             and n["language"] == candidate["language"] for n in record["names"]):
    return None
  evidence = record["evidence"]
  if source["id"] == SCIDB_SOURCE["id"]:
    if (source != SCIDB_SOURCE or candidate["kind"] != "alias"
        or evidence.get("family") != item.get("taxonomy", {}).get("family")
        or evidence.get("genus") != item.get("taxonomy", {}).get("genus")):
      return None
    attribution = {
      **source,
      "title": "中国外来植物名录数据集 · 林秦文、崔晓、马金双（2022，CC BY 4.0；别名提取）",
      "recordId": record["id"],
    }
  elif source["id"] == "wikidata":
    if (flags or source.get("license") != "CC0-1.0"
        or candidate["kind"] != "vernacular_statement"
        or not evidence.get("references") or not evidence.get("revision")):
      return None
    attribution = {
      "title": "Wikidata 俗名声明（引自《中国植物志》，CC0）",
      "url": evidence["sourceUrl"], "version": str(evidence["revision"]),
      "license": "CC0-1.0",
      "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
      "recordId": record["id"], "references": evidence["references"],
    }
  else:
    return None
  return {
    **attribution, "sourceName": name, "language": candidate["language"],
    "matchedScientificName": item["scientificName"], "matchType": "exact",
    "acceptancePolicy": POLICY, "candidateId": candidate["id"],
  }


def apply_aliases(catalog: Path, candidates: Path, output: Path) -> dict:
  if output.exists() or output.is_symlink():
    raise ValueError("Output already exists; choose a new path.")
  catalog_hash, candidate_hash = digest(catalog), digest(candidates)
  with closing(sqlite3.connect(candidates.resolve().as_uri() + "?mode=ro", uri=True)) as pending:
    pending.row_factory = sqlite3.Row
    manifest = json.loads(pending.execute("SELECT data FROM metadata").fetchone()[0])
    if manifest.get("schemaVersion") != 1 or manifest.get("purpose") != "pending_name_enrichment":
      raise ValueError("Unsupported name candidate dataset.")
    sources = {r["id"]: json.loads(r["data"]) for r in pending.execute("SELECT * FROM sources")}
    rows = [dict(row) for row in pending.execute("""
      SELECT c.*, r.source_id, r.evidence FROM candidates c
      JOIN records r ON r.id=c.record_id ORDER BY c.id
    """)]
  output.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix="catalog-aliases-", dir=output.parent)
  os.close(descriptor)
  counts = Counter()
  try:
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as original:
      with closing(sqlite3.connect(temporary)) as db, db:
        original.backup(db)
        metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
        previous = metadata.get("aliasEnrichment", {})
        if catalog_hash != manifest["catalogSha256"] and not (
            previous.get("candidateSha256") == candidate_hash
            and previous.get("baseCatalogSha256") == manifest["catalogSha256"]
            and previous.get("policy") == POLICY):
          raise ValueError("Candidates belong to another catalog snapshot; rebuild them first.")
        changed = set()
        for candidate in rows:
          row = db.execute("SELECT data FROM species WHERE id=?", (candidate["species_id"],)).fetchone()
          if not row:
            counts["skipped"] += 1
            continue
          item = json.loads(row[0])
          source = alias_source(
            candidate, json.loads(candidate["evidence"]), sources[candidate["source_id"]], item,
          )
          if not source:
            counts["skipped"] += 1
            continue
          aliases, name_sources = item.setdefault("chineseAliases", []), item.setdefault("nameSources", [])
          added = False
          if candidate["name"] not in aliases and candidate["name"] != item.get("chineseName"):
            aliases.append(candidate["name"])
            counts["aliasesAdded"] += 1
            added = True
          if not any(s.get("candidateId") == candidate["id"] for s in name_sources):
            name_sources.append(source)
            counts["sourcesAdded"] += 1
            added = True
          if not added:
            counts["alreadyApplied"] += 1
            continue
          # Preserve original search terms, including source English names and taxonomy.
          search = db.execute("SELECT search_text FROM species WHERE id=?", (item["id"],)).fetchone()[0]
          db.execute("UPDATE species SET data=?,search_text=? WHERE id=?", (
            json.dumps(item, ensure_ascii=False), search + " " + candidate["name"], item["id"],
          ))
          changed.add(item["id"])
        report = {
          "policy": POLICY, "baseCatalogSha256": manifest["catalogSha256"],
          "candidateSha256": candidate_hash, "changedSpecies": len(changed),
          "aliasesAdded": counts["aliasesAdded"], "sourcesAdded": counts["sourcesAdded"],
          "skipped": counts["skipped"], "alreadyApplied": counts["alreadyApplied"],
        }
        if changed:
          metadata["aliasEnrichment"] = {
            "policy": POLICY, "baseCatalogSha256": manifest["catalogSha256"],
            "candidateSha256": candidate_hash,
          }
          db.execute("UPDATE metadata SET data=?", (json.dumps(metadata, ensure_ascii=False),))
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
          raise ValueError("Enriched catalog failed integrity check.")
    if digest(catalog) != catalog_hash or digest(candidates) != candidate_hash:
      raise ValueError("Input changed during build.")
    os.link(temporary, output)
  finally:
    Path(temporary).unlink(missing_ok=True)
  return report


def reveal_sciencedb(
    catalog: Path, workbook: Path, output: Path, *, expected_count: int | None = None,
) -> dict:
  """Fill missing Chinese names from the pinned workbook and reveal those species only."""
  if output.exists() or output.is_symlink():
    raise ValueError("Output already exists; choose a new path.")
  index = {}
  for row in scidb_rows(workbook):
    scientific, chinese = row["学名-scientific name"], row["中文名-Chinese name"]
    if not BINOMIAL.fullmatch(scientific) or not HAN.search(chinese) or len(chinese) > 80:
      continue
    if scientific in index:
      raise ValueError("Ambiguous scientific name in ScienceDB workbook.")
    index[scientific] = row
  catalog_hash = digest(catalog)
  output.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix="catalog-primary-", dir=output.parent)
  os.close(descriptor)
  changed, taxonomy_differences = 0, 0
  try:
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as original:
      with closing(sqlite3.connect(temporary)) as db, db:
        original.backup(db)
        metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
        rows = db.execute("""
          SELECT id,scientific_name,data,search_text FROM species
          WHERE is_visible=0 AND has_chinese_name=0
        """).fetchall()
        for identifier, scientific, data, search in rows:
          source_row = index.get(scientific)
          if source_row is None:
            continue
          item = json.loads(data)
          if item.get("chineseName") or item.get("isVisible"):
            continue  # Never overwrite an existing name or another visibility policy.
          if item["id"] != identifier or item["scientificName"] != scientific:
            raise ValueError("Catalog identity fields disagree.")
          chinese = source_row["中文名-Chinese name"]
          source_taxonomy = {
            "family": source_row["科名-Family"], "genus": source_row["属名-Genus"],
          }
          differences = [
            rank for rank, name in source_taxonomy.items()
            if name != (item.get("taxonomy") or {}).get(rank)
          ]
          taxonomy_differences += bool(differences)
          item.update({
            "commonName": chinese, "chineseName": chinese,
            "chineseNameOrigin": "sciencedb", "isVisible": True,
          })
          item.setdefault("nameSources", []).append({
            **SCIDB_SOURCE,
            "title": "中国外来植物名录数据集 · 林秦文、崔晓、马金双（2022，CC BY 4.0；中文名提取）",
            "modifications": "Extracted source Chinese name; existing catalog taxonomy unchanged.",
            "recordId": f"sciencedb-01711-v1:row-{source_row['rowNumber']}",
            "sourceName": chinese, "matchedScientificName": scientific,
            "matchType": "exact", "acceptancePolicy": PRIMARY_POLICY,
            "sourceTaxonomy": source_taxonomy, "taxonomyDifferences": differences,
          })
          db.execute("""
            UPDATE species SET data=?,search_text=?,has_chinese_name=1,is_visible=1 WHERE id=?
          """, (json.dumps(item, ensure_ascii=False), search + " " + chinese, identifier))
          changed += 1
        if expected_count is not None and changed != expected_count:
          raise ValueError(f"Expected {expected_count} newly visible species, found {changed}.")
        total, chinese_count, visible = db.execute(
          "SELECT count(*),sum(has_chinese_name),sum(is_visible) FROM species"
        ).fetchone()
        if changed:
          metadata.update({
            "chineseNameCount": chinese_count, "visiblePlantCount": visible,
            "hiddenPlantCount": total - visible,
            "primaryNameEnrichment": {
              "policy": PRIMARY_POLICY, "source": SCIDB_SOURCE,
              "baseCatalogSha256": catalog_hash, "revealedSpecies": changed,
              "taxonomyDifferenceCount": taxonomy_differences,
            },
          })
          db.execute("UPDATE metadata SET data=?", (json.dumps(metadata, ensure_ascii=False),))
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
          raise ValueError("Updated catalog failed integrity check.")
        report = {
          "policy": PRIMARY_POLICY, "revealedSpecies": changed,
          "visiblePlantCount": visible, "hiddenPlantCount": total - visible,
          "chineseNameCount": chinese_count, "taxonomyDifferenceCount": taxonomy_differences,
          "baseCatalogSha256": catalog_hash, "sourceSha256": SCIDB_SOURCE["snapshotSha256"],
        }
    if digest(catalog) != catalog_hash:
      raise ValueError("Input catalog changed during build.")
    os.link(temporary, output)
  finally:
    Path(temporary).unlink(missing_ok=True)
  return report
