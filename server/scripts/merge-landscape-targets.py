"""Merge user name lists into versioned species targets, without changing live training."""

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research"
BASE = ROOT / "server/.artifacts/china-landscape-from-scratch-20260924/targets.json"
CATALOG = ROOT / "server/.cache/catalog/treeoflife-10m/catalog.sqlite3"
INPUT = DOCS / "ornamental-plants-120.txt"
NOTES = DOCS / "ornamental-plants-120-mapping-notes.json"
OUTPUT = DOCS / "landscape-targets-merged-20260924"
OLD_LIST = "landscape-plants-200"
NEW_LIST = "ornamental-plants-120"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(base, names, notes, records):
    by_id = {r["speciesId"]: r for r in records}
    by_name, by_scientific, previous = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in records:
        for name in [row["chineseName"], row["displayName"], *row["aliases"]]:
            if name:
                by_name[name].add(row["speciesId"])
        by_scientific[row["scientificName"]].add(row["speciesId"])
    classes = deepcopy(base["classes"])
    if len({r["speciesId"] for r in classes}) != len(classes):
        raise ValueError("Duplicate IDs in original target.")
    original_ids = {r["speciesId"] for r in classes}
    merged = {r["speciesId"]: r for r in classes}
    for row in classes:
        if row["scientificName"] != by_id[row["speciesId"]]["scientificName"]:
            raise ValueError("Original target identity differs from catalog.")
        for source in row["sourceInputs"]:
            source["inputList"] = OLD_LIST
            previous[source["inputName"]].add(row["speciesId"])
    unresolved = [
        {**deepcopy(row), "inputList": OLD_LIST} for row in base["unresolvedInputs"]
    ]
    previous_unresolved = {r["inputName"]: r["note"] for r in unresolved}
    mappings = []
    for position, name in enumerate(names, 1):
        candidate_ids = by_name[name] | previous[name]
        note, sources, method = "", [], "catalog-exact-or-previous-mapping"
        blocked = name in notes["needsReview"] or name in previous_unresolved
        reviewed = notes.get("reviewedMappings", {}).get(name)
        if reviewed:
            if not reviewed.get("sources") or not reviewed.get("note"):
                raise ValueError("Reviewed mappings need sources and a reason.")
            candidate_ids = set(reviewed.get("candidateSpeciesIds", []))
            if not candidate_ids <= set(by_id):
                raise ValueError("Reviewed candidate ID missing from catalog.")
            note, sources, method = reviewed["note"], reviewed["sources"], "source-reviewed"
            blocked = reviewed["status"] != "linked"
            if not blocked:
                key = reviewed["speciesId"]
                if key not in by_id or (
                    by_id[key]["scientificName"] != reviewed["scientificName"]
                ):
                    raise ValueError("Reviewed identity differs from catalog.")
                if not reviewed.get("chineseName"):
                    raise ValueError("Reviewed mapping needs a sourced Chinese name.")
                candidate_ids = {key}
        elif blocked:
            note = notes["needsReview"].get(name) or previous_unresolved[name]
            sources = notes.get("reviewSources", {}).get(name, [])
        elif name in notes["scientificNameLinks"]:
            link = notes["scientificNameLinks"][name]
            candidate_ids = by_scientific[link["scientificName"]]
            note, sources, method = link["note"], link["sources"], "scoped-scientific-link"
        record = deepcopy(by_id[next(iter(candidate_ids))]) if len(candidate_ids) == 1 else None
        localization_required = bool(record and (
            not record["chineseName"] or not record["isVisible"]
        ))
        if reviewed and not blocked and record and not record["chineseName"]:
            record["chineseName"] = reviewed["chineseName"]
        linked = (
            not blocked and record is not None
            and bool(record["chineseName"]) and (record["isVisible"] or reviewed is not None)
        )
        source = {"inputList": NEW_LIST, "position": position, "inputName": name}
        entry = {
            **source, "status": "linked" if linked else "needs-review",
            "matchMethod": method,
            "speciesId": record["speciesId"] if linked else None,
            "scientificName": record["scientificName"] if linked else None,
            "chineseName": record["chineseName"] if linked else None,
            "alreadyInOriginal162": record["speciesId"] in original_ids if linked else False,
            "candidates": [by_id[key] for key in sorted(candidate_ids)],
            "note": note, "sources": sources,
            "catalogLocalizationRequired": bool(linked and localization_required),
        }
        if linked:
            key = record["speciesId"]
            if key not in merged:
                merged[key] = {
                    k: record[k] for k in ("speciesId", "scientificName", "chineseName")
                }
                merged[key]["sourceInputs"] = []
                if localization_required:
                    merged[key]["catalogLocalizationRequired"] = True
                    merged[key]["chineseNameSources"] = sources
                classes.append(merged[key])
            merged[key]["sourceInputs"].append(source)
        else:
            entry["note"] = note or (
                "No unique visible Chinese-named catalog match; do not guess species."
            )
            unresolved.append({
                **source, "status": "needs-review", "note": entry["note"],
                "candidateSpeciesIds": sorted(candidate_ids),
            })
        mappings.append(entry)
    counts = Counter(row["status"] for row in mappings)
    summary = {
        "originalSpecies": len(base["classes"]), "additionalInputNames": len(names),
        "linkedAdditionalNames": counts["linked"],
        "overlappingAdditionalNames": sum(r["alreadyInOriginal162"] for r in mappings),
        "newDistinctSpecies": len(classes) - len(base["classes"]),
        "mergedDistinctSpecies": len(classes),
        "pendingAdditionalNames": counts["needs-review"],
        "pendingOriginalNames": len(base["unresolvedInputs"]),
        "pendingDistinctNames": len({r["inputName"] for r in unresolved}),
        "catalogLocalizationRequiredSpecies": sum(
            bool(r.get("catalogLocalizationRequired")) for r in classes
        ),
    }
    result = {
        "createdAt": datetime.now(UTC).isoformat(), "level": "species",
        "status": "planned-merged-targets-not-trained-support",
        "inputRecordCount": base["inputRecordCount"] + len(names),
        "linkedInputCount": base["linkedInputCount"] + counts["linked"],
        "resolvedUniqueSpeciesCount": len(classes),
        "unresolvedInputCount": len(unresolved),
        "summary": summary, "classes": classes, "unresolvedInputs": unresolved,
        "additionalInputMappings": mappings, "trainedSupportDeclared": False,
        "sourceVerificationNote": notes.get("sourceVerification"),
        "databaseWritten": False, "runningCollectionOrTrainingChanged": False,
        "unresolvedMayNotBeSilentlyDropped": True,
    }
    assert result["inputRecordCount"] == result["linkedInputCount"] + len(unresolved)
    assert original_ids <= set(merged)
    assert len({r["scientificName"] for r in classes}) == len(classes)
    assert len(mappings) == len(names)
    assert sum(len(r["sourceInputs"]) for r in classes) == result["linkedInputCount"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=NOTES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    destinations = [args.output.with_suffix(ext) for ext in (".json", ".csv", ".txt")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Versioned merge output already exists; do not overwrite.")
    base_digest = digest(BASE)
    base = json.loads(BASE.read_text())
    names = INPUT.read_text().splitlines()
    if len(names) != 120 or len(set(names)) != 120:
        raise ValueError("Expected all 120 distinct user-provided names.")
    notes = json.loads(args.notes.read_text())
    with sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True) as db:
        records = [
            {
                "speciesId": key, "scientificName": scientific, "chineseName": chinese,
                "aliases": json.loads(aliases), "displayName": display, "isVisible": bool(visible),
            }
            for key, scientific, chinese, aliases, display, visible in db.execute(
                "SELECT c.id,c.scientific_name,z.chinese_name,z.aliases,z.display_name,z.is_visible "
                "FROM species_core c JOIN species_zh z ON z.species_id=c.id"
            )
        ]
    result = merge(base, names, notes, records)
    result["sourceFiles"] = [
        {"path": str(p.resolve().relative_to(ROOT)), "sha256": digest(p)}
        for p in (BASE, INPUT, args.notes)
    ]
    result["catalogSha256"] = digest(CATALOG)
    with destinations[0].open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with destinations[1].open("x") as stream:
        writer = csv.writer(stream)
        writer.writerow(["species_id", "scientific_name", "chinese_name", "new_to_162",
                         "source_names"])
        original = {r["speciesId"] for r in base["classes"]}
        for row in result["classes"]:
            writer.writerow([
                row["speciesId"], row["scientificName"], row["chineseName"],
                row["speciesId"] not in original,
                "；".join(dict.fromkeys(s["inputName"] for s in row["sourceInputs"])),
            ])
    with destinations[2].open("x") as stream:
        stream.write("\n".join(r["chineseName"] for r in result["classes"]) + "\n")
    assert digest(BASE) == base_digest, "Frozen training targets must remain unchanged."
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("Pending:", "、".join(
        r["inputName"] for r in result["additionalInputMappings"] if r["status"] != "linked"
    ))


if __name__ == "__main__":
    main()
