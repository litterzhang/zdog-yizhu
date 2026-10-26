"""Separate public A rehearsal and B addition pools; never publish planned support."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from yizhu_server.everyday_training_data import TARGETS, TaxonResolver
from yizhu_server.plantnet_transfer import verified_photo
from yizhu_server.training_classifier import validate_samples
from yizhu_server.training_data import (
    ROOT,
    PublicClient,
    collect,
    dataset_lock,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)

DEFAULT_SETS = ROOT.parent / "docs/research/plantnet-plant-sets-20260923/sets.json"
DEFAULT_DATA = ROOT / ".cache/training/plantnet-china-additions-883"
DEFAULT_RETAINED_DATA = ROOT / ".cache/training/plantnet-china-retained-420"
POLICY = "plantnet-chinese-additions-only-v1"
RETAINED_POLICY = "plantnet-chinese-retained-rehearsal-v1"


def prioritize(document, identifiers):
    """Change scheduling only; never mutate the frozen target or add unsupported labels."""
    by_id = {row["speciesId"]: row for row in document["species"]}
    if len(set(identifiers)) != len(identifiers) or any(key not in by_id for key in identifiers):
        raise ValueError("Priority IDs must be distinct members of the existing target set.")
    selected = set(identifiers)
    return {
        **document,
        "species": [by_id[key] for key in identifiers]
        + [row for row in document["species"] if row["speciesId"] not in selected],
    }


def union_targets(sets):
    existing = sets["pretrainedSupported"]
    additions = sets["finetuneAdditions"]
    if {r["speciesId"] for r in existing} & {r["speciesId"] for r in additions}:
        raise ValueError("Existing and additional species sets overlap.")
    rows = [
        {**row, "classRole": role}
        for role, members in (("retained-pretrained", existing), ("finetune-addition", additions))
        for row in members
    ]
    if (
        len({r["speciesId"] for r in rows}) != len(rows)
        or len({r["scientificName"] for r in rows}) != len(rows)
        or any(not r.get("chineseName") for r in rows)
    ):
        raise ValueError("Union requires distinct species IDs/names and Chinese names.")
    counts = sets["counts"]
    if (
        len(existing) != counts["pretrainedChineseSupported"]
        or len(additions) != counts["finetuneAdditions"]
        or len(rows) != counts["combinedTarget"]
    ):
        raise ValueError("Exported species-set counts do not agree.")
    # Collect regional everyday candidates before retained species outside that candidate list.
    return sorted(rows, key=lambda r: (not r["inEverydayCandidates"], r["scientificName"]))


def prepare(root, sets_path, *, scope="additions"):
    if scope not in {"additions", "retained"}:
        raise ValueError("Unknown collection scope.")
    retained = scope == "retained"
    policy = RETAINED_POLICY if retained else POLICY
    require_dataset_active(root)
    digest = sha256(sets_path)
    destination = root / "species.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        if result.get("policy") != policy or result.get("plantSetsSha256") != digest:
            raise ValueError("Dataset scope differs; use a new directory.")
        return result
    sets = json.loads(sets_path.read_text())
    union = union_targets(sets)
    members = [
        row for row in union
        if row["classRole"] == ("retained-pretrained" if retained else "finetune-addition")
    ]
    result = {
        "createdAt": now(), "policy": policy, "plantSetsSha256": digest,
        "catalogSha256": sets["catalogSha256"],
        "species": members,
        "reservedSourceNames": {row["scientificName"]: row["speciesId"] for row in union},
        "requestedClassCount": len(members), "splitTargets": TARGETS,
        "combinedSolutionTargetCount": sets["counts"]["combinedTarget"],
        "retainedPretrainedIds": [r["speciesId"] for r in sets["pretrainedSupported"]],
        "additionalSpeciesIds": [r["speciesId"] for r in sets["finetuneAdditions"]],
        "sourceNamePolicy": "exact-active-name-or-explicit-listed-scientific-synonym",
        "limitations": [
            (
                "A-only public rehearsal/regression pool; no new classes or support expansion."
                if retained else
                "B collection only; existing A is skipped, including the 40 overlapping candidates."
            ),
            *([
                "Only train-split A photos may constrain B outputs in explicit replay trials.",
                "A validation/test photos remain held out; not independent final field validation.",
            ] if retained else []),
            "Collecting/training B does not publish A union B support; verify actual model first.",
            "Chinese-visible catalog snapshot, not a national commonness ranking.",
            "Global research-grade public photos, not expert-audited or China-only field photos.",
            "Source-listed scientific synonyms preserve catalog IDs/names; catalog is not rewritten.",
            "Unresolved identities, duplicate source taxa and insufficient photos remain gaps.",
            "Only photo-level CC0/CC BY; pretrained overlap is not ruled out.",
        ],
    }
    write_json(destination, result)
    return result


def seed_public_cache(root, source, document):
    """Copy only eligible public samples; caller holds locks on both datasets."""
    require_dataset_active(root)
    require_dataset_active(source)
    if root.resolve() == source.resolve():
        raise ValueError("Use separate source and additions dataset directories.")
    source_digest = sha256(source / "species.json")
    report_path = root / "seed-import.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["sourceSpeciesSha256"] != source_digest
            or report["targetSpeciesSha256"] != sha256(root / "species.json")
        ):
            raise ValueError("Seed source differs from the completed import.")
        return report
    if (root / "collection.json").exists() or load_samples(root):
        raise ValueError("Seed only before collection starts; do not overwrite existing samples.")
    wanted = {r["speciesId"]: r for r in document["species"]}
    original_rows = load_samples(source)
    rows = [r for r in original_rows if r["speciesId"] in wanted]
    validate_samples(rows, document["species"])
    selected_ids = {r["speciesId"] for r in rows}
    mappings = {}
    for path in (source / "taxon-matches").glob("*.json"):
        record = json.loads(path.read_text())
        if record["speciesId"] in selected_ids and record["status"] == "matched":
            mappings[record["speciesId"]] = record
    if selected_ids != set(mappings) or len({
        r["inatTaxonId"] for r in mappings.values()
    }) != len(mappings):
        raise ValueError("Seed photos need unique source taxon mappings.")
    by_species = {}
    for row in rows:
        photo = verified_photo(source, row)
        if shutil.disk_usage(root).free < 4 * 1024**3:
            raise OSError("Keep at least 4 GiB free while importing public photos.")
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verified_photo(root, row)
        else:
            temporary = target.with_suffix(".seed-tmp")
            shutil.copyfile(photo, temporary)
            temporary.replace(target)
            verified_photo(root, row)
        by_species.setdefault(row["speciesId"], []).append(row)
    for identifier, samples in by_species.items():
        record = mappings[identifier]
        if record["scientificName"] != wanted[identifier]["scientificName"]:
            raise ValueError("Seed taxonomy mapping changed.")
        filename = hashlib.sha256(identifier.encode()).hexdigest() + ".json"
        write_json(root / "taxon-matches" / filename, record)
        counts = Counter(r["split"] for r in samples)
        write_json(root / "samples" / f"{record['inatTaxonId']}.json", {
            "samples": samples, "counts": dict(counts),
            "complete": all(counts[k] >= value for k, value in TARGETS.items()),
        })
    report = {
        "createdAt": now(), "sourceSpeciesSha256": source_digest,
        "targetSpeciesSha256": sha256(root / "species.json"),
        "copiedImages": len(rows), "copiedSpecies": len(selected_ids),
        "excludedImages": len(original_rows) - len(rows),
        "sourcePreserved": True,
    }
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seed", "collect"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--scope", choices=("additions", "retained"), default="additions")
    parser.add_argument("--sets", type=Path, default=DEFAULT_SETS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--priority-species", nargs="+", default=[],
                        help="Collect these existing scope IDs first; does not change membership")
    parser.add_argument("--class-limit", type=int,
                        help="Bound this collection pass, without shrinking the fixed target set")
    args = parser.parse_args()
    if args.data is None:
        args.data = DEFAULT_RETAINED_DATA if args.scope == "retained" else DEFAULT_DATA
    if args.action == "seed" and args.source is None:
        parser.error("seed requires --source (existing public dataset)")
    if args.priority_species and args.action != "collect":
        parser.error("--priority-species is only valid for collect")
    if args.class_limit is not None and (args.action != "collect" or args.class_limit < 1):
        parser.error("--class-limit requires collect and a positive limit")
    with dataset_lock(args.data):
        document = prepare(args.data, args.sets, scope=args.scope)
        print(f"Fixed {args.scope} scope: {len(document['species'])} species", flush=True)
        if args.action == "seed":
            with dataset_lock(args.source):
                print(json.dumps(seed_public_cache(args.data, args.source, document), indent=2))
        elif args.action == "collect":
            client = PublicClient()
            scheduled = prioritize(document, args.priority_species)
            write_json(args.data / "run-order.json", {
                "updatedAt": now(), "prioritySpeciesIds": args.priority_species,
                "targetSpeciesSha256": sha256(args.data / "species.json"),
                "targetCount": len(scheduled["species"]),
                "passClassLimit": min(args.class_limit or len(scheduled["species"]),
                                      len(scheduled["species"])),
                "note": "Only processing order differs; frozen membership/labels are unchanged.",
            })
            try:
                collect(
                    args.data, client, scheduled,
                    class_limit=min(args.class_limit or len(scheduled["species"]),
                                    len(scheduled["species"])),
                    split_targets=TARGETS, max_pages=20,
                    resolve_species=TaxonResolver(
                        args.data, client, allow_source_names=True,
                        target_names=document["reservedSourceNames"],
                    ),
                )
            except Exception as exc:
                progress = args.data / "progress.json"
                previous = json.loads(progress.read_text()) if progress.exists() else {}
                write_json(progress, {
                    **previous, "updatedAt": now(), "stage": "stopped",
                    "errorType": type(exc).__name__, "note": "Inspect log before resuming.",
                })
                raise


if __name__ == "__main__":
    main()
