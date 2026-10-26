"""Reveal reviewed target species that have explicit sourced Chinese names."""

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from yizhu_server.name_enrichment import HAN, digest
from yizhu_server.training_data import now, sha256, write_json

POLICY = "target249-reviewed-sourced-primary-v1"


def localize(catalog, targets_path, species_ids, output):
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new localized catalog path.")
    targets_document = json.loads(targets_path.read_text())
    targets = {row["speciesId"]: row for row in targets_document["classes"]}
    if (
        len(species_ids) != len(set(species_ids)) or not species_ids
        or not set(species_ids) <= set(targets)
    ):
        raise ValueError("Require distinct reviewed target species IDs.")
    base_hash = digest(catalog)
    target_hash = sha256(targets_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="catalog-target249-localized-", suffix=".sqlite3", dir=output.parent,
    )
    os.close(descriptor)
    changed = []
    try:
        with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(temporary)) as db, db:
                source.backup(db)
                metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
                for sid in species_ids:
                    target = targets[sid]
                    chinese = target.get("chineseName")
                    source_urls = target.get("chineseNameSources", [])
                    if (
                        not isinstance(chinese, str) or not HAN.search(chinese)
                        or not isinstance(source_urls, list) or not source_urls
                        or any(
                            not isinstance(url, str)
                            or not url.startswith(("https://", "http://"))
                            for url in source_urls
                        )
                    ):
                        raise ValueError("Target Chinese name lacks explicit source evidence.")
                    row = db.execute("""
                      SELECT c.scientific_name,z.chinese_name,z.aliases,z.name_sources,
                        z.search_text,z.is_visible
                      FROM species_core c JOIN species_zh z ON z.species_id=c.id
                      WHERE c.id=?
                    """, (sid,)).fetchone()
                    if row is None or row[0] != target["scientificName"]:
                        raise ValueError("Target identity differs from the catalog.")
                    if row[1] is not None or row[5] != 0:
                        raise ValueError("Localization only fills hidden species with no Chinese name.")
                    aliases = json.loads(row[2])
                    for source_input in target.get("sourceInputs", []):
                        value = source_input.get("inputName")
                        if (
                            isinstance(value, str) and HAN.search(value)
                            and value != chinese and value not in aliases
                        ):
                            aliases.append(value)
                    name_sources = json.loads(row[3])
                    for index, url in enumerate(source_urls):
                        name_sources.append({
                            "title": "249种中国常见园林与观赏植物审核名源",
                            "url": url, "sourceName": chinese,
                            "matchedScientificName": target["scientificName"],
                            "matchType": "reviewed-exact-target",
                            "acceptancePolicy": POLICY,
                            "targetManifestSha256": target_hash,
                            "sourcePosition": index,
                        })
                    search = " ".join(dict.fromkeys([
                        row[4], chinese, *aliases,
                    ]))
                    db.execute("""
                      UPDATE species_zh
                      SET chinese_name=?,aliases=?,display_name=?,name_origin=?,
                        name_sources=?,search_text=?,is_visible=1
                      WHERE species_id=?
                    """, (
                        chinese, json.dumps(aliases, ensure_ascii=False), chinese, POLICY,
                        json.dumps(name_sources, ensure_ascii=False), search, sid,
                    ))
                    changed.append({
                        "speciesId": sid, "scientificName": target["scientificName"],
                        "chineseName": chinese, "aliases": aliases, "sources": source_urls,
                    })
                total, chinese_count, visible = db.execute(
                    "SELECT count(*),sum(has_chinese_name),sum(is_visible) FROM species"
                ).fetchone()
                metadata.update({
                    "chineseNameCount": chinese_count, "visiblePlantCount": visible,
                    "hiddenPlantCount": total - visible,
                    "target249Localization": {
                        "policy": POLICY, "createdAt": now(),
                        "baseCatalogSha256": base_hash,
                        "targetManifestSha256": target_hash,
                        "revealedSpecies": len(changed),
                        "speciesIds": species_ids,
                    },
                })
                db.execute(
                    "UPDATE metadata SET data=?",
                    (json.dumps(metadata, ensure_ascii=False),),
                )
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Localized catalog failed integrity check.")
        if digest(catalog) != base_hash:
            raise ValueError("Catalog changed while localization snapshot was built.")
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "createdAt": now(), "policy": POLICY,
        "baseCatalogSha256": base_hash, "targetManifestSha256": target_hash,
        "outputSha256": digest(output), "revealedSpecies": len(changed),
        "species": changed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--species-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = localize(
        args.catalog.resolve(), args.targets.resolve(), args.species_id,
        args.output.resolve(),
    )
    write_json(args.report.resolve(), report)
    print(json.dumps({
        "revealedSpecies": report["revealedSpecies"],
        "outputSha256": report["outputSha256"],
    }))


if __name__ == "__main__":
    main()
