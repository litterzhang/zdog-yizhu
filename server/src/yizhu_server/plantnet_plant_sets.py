"""Read-only export of Pl@ntNet's Chinese-supported set and proposed additions."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from yizhu_server.catalog import CATALOG_PATH, PlantCatalog
from yizhu_server.plantnet_pretrained import (
    DEFAULT_DIRECTORY,
    class_records,
    read_class_map,
    verify,
)
from yizhu_server.training_data import ROOT, now, sha256, write_json

DEFAULT_TARGETS = ROOT / ".cache/training/china-everyday-972/species.json"


def pretrained_support_contract(sets, manifest):
    """Only A becomes capability metadata. Planned B and the 40 overlapping targets do not."""
    if sets["sourceFiles"] != manifest["files"]:
        raise ValueError("Plant-set export is not bound to this pretrained checkpoint.")
    members = sets["pretrainedSupported"]
    if (
        len(members) != sets["counts"]["pretrainedChineseSupported"]
        or len({row["speciesId"] for row in members}) != len(members)
        or any(not row.get("chineseName") or not row.get("sourceClasses") for row in members)
    ):
        raise ValueError("Invalid pretrained supported set.")
    return {
        "schemaVersion": 1, "solutionId": "plantnet-300k-resnet50",
        "status": "pretrained", "sourceFiles": manifest["files"],
        "supportedSpecies": members,
        "updateRule": "Expand only from an actually trained model artifact, never from a target list.",
        "accuracyGuarantee": False, "productionChanged": False,
    }


def register_pretrained_support(directory, sets_path, *, assets=DEFAULT_DIRECTORY):
    from yizhu_server.solution_metadata import immutable

    manifest = verify(assets, "resnet50")
    result = pretrained_support_contract(json.loads(sets_path.read_text()), manifest)
    class_map = read_class_map(assets)
    for row in result["supportedSpecies"]:
        for source in row["sourceClasses"]:
            index = source["classIndex"]
            if (
                type(index) is not int or not 0 <= index < len(class_map)
                or class_map[index] != (source["sourceSpeciesId"], source["sourceScientificName"])
            ):
                raise ValueError("Support member is not an actual pretrained output class.")
    immutable(directory / "contracts/plantnet-300k-resnet50.json", result)
    return {"supportedPlantCount": len(result["supportedSpecies"]), "status": result["status"]}


def build_sets(records, chinese_rows, candidates):
    """Language-table membership alone is insufficient: migration retains empty locale rows."""
    ids = [row["speciesId"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate IDs.")
    usable = {
        key for key, row in chinese_rows.items()
        if row["has_chinese_name"] and row["is_visible"]
    }
    mapped = {}
    for row in records:
        identifier = row.get("catalogSpeciesId")
        if identifier in chinese_rows:
            mapped.setdefault(identifier, []).append({
                key: row[key]
                for key in ("classIndex", "sourceSpeciesId", "sourceScientificName")
            })
    existing = set(mapped) & usable
    targets = set(ids)
    eligible = targets & usable
    additions = eligible - existing
    candidate_by_id = {row["speciesId"]: row for row in candidates}

    def items(identifiers):
        output = []
        for identifier in sorted(identifiers, key=lambda k: chinese_rows[k]["scientific_name"]):
            row = chinese_rows[identifier]
            output.append({
                "speciesId": identifier, "scientificName": row["scientific_name"],
                "chineseName": row["chinese_name"], "chineseAliases": json.loads(row["aliases"]),
                "sourceClasses": mapped.get(identifier, []),
                "inEverydayCandidates": identifier in targets,
                "candidateId": candidate_by_id.get(identifier, {}).get("candidateId"),
            })
        return output

    excluded = [{
        "speciesId": identifier,
        "scientificName": candidate_by_id[identifier]["scientificName"],
        "reason": (
            "no-species-zh-row" if identifier not in chinese_rows
            else "no-chinese-name" if not chinese_rows[identifier]["has_chinese_name"]
            else "hidden"
        ),
    } for identifier in sorted(targets - eligible)]
    return {
        "counts": {
            "pretrainedOutputClasses": len(records),
            "pretrainedMappedToZhRowsRegardlessOfName": len(mapped),
            "pretrainedChineseSupported": len(existing),
            "candidateSpecies": len(targets),
            "candidateSpeciesInZhTable": len(targets & set(chinese_rows)),
            "candidateChineseVisible": len(eligible),
            "candidateAlreadySupported": len(eligible & existing),
            "finetuneAdditions": len(additions),
            "candidateExcluded": len(excluded),
            "combinedTarget": len(existing | additions),
            "existingOutsideEverydayCandidates": len(existing - targets),
        },
        "pretrainedSupported": items(existing),
        "finetuneAdditions": items(additions),
        "candidateAlreadySupported": items(eligible & existing),
        "excludedCandidates": excluded,
    }


def export(output, *, catalog=CATALOG_PATH, targets=DEFAULT_TARGETS, assets=DEFAULT_DIRECTORY):
    if output.exists():
        raise ValueError("Export directory exists; use a new snapshot path.")
    # Check the labels really belong to our downloaded checkpoint; do not load a neural model.
    manifest = verify(assets, "resnet50")
    records = class_records(read_class_map(assets), PlantCatalog(catalog))
    with sqlite3.connect(f"file:{catalog.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        chinese_rows = {
            row["species_id"]: dict(row) for row in db.execute(
                "SELECT c.scientific_name,z.species_id,z.chinese_name,z.aliases,"
                "z.has_chinese_name,z.is_visible FROM species_core c "
                "JOIN species_zh z ON z.species_id=c.id"
            )
        }
    candidates = json.loads(targets.read_text())["species"]
    result = {
        "createdAt": now(), "solution": "plantnet-300k-resnet50",
        "sourceVariant": "official-1081-class-pretrained-not-V2",
        "catalogSha256": sha256(catalog), "targetsSha256": sha256(targets),
        "sourceFiles": manifest["files"],
        "eligibility": "species_zh.has_chinese_name=1 AND species_zh.is_visible=1",
        "matching": "Existing conservative exact scientific-name mapping; no guessed synonyms",
        "setRule": "A=pretrained mapped Chinese-visible species; B=Chinese-visible candidates minus A",
        "finetuneRule": "Train/evaluate A union B to retain A; do not replace A with B-only head",
        "accuracyGuarantee": False, "productionChanged": False,
        **build_sets(records, chinese_rows, candidates),
    }
    output.mkdir(parents=True)
    write_json(output / "sets.json", result)
    fields = ["speciesId", "chineseName", "chineseAliases", "scientificName",
              "sourceClassIndices", "inEverydayCandidates", "candidateId"]
    for key, filename in (
        ("pretrainedSupported", "pretrained-supported.csv"),
        ("finetuneAdditions", "finetune-additions.csv"),
        ("candidateAlreadySupported", "candidate-already-supported.csv"),
    ):
        with (output / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in result[key]:
                writer.writerow({
                    **{key: row[key] for key in fields if key not in {
                        "chineseAliases", "sourceClassIndices",
                    }},
                    "chineseAliases": " / ".join(row["chineseAliases"]),
                    "sourceClassIndices": " / ".join(str(c["classIndex"]) for c in row["sourceClasses"]),
                })
    with (output / "excluded-candidates.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["speciesId", "scientificName", "reason"])
        writer.writeheader()
        writer.writerows(result["excludedCandidates"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
