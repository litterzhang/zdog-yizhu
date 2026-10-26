"""Collect only approved landscape species; reuse public photos, never model features."""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from yizhu_server.everyday_training_data import TARGETS, TaxonResolver
from yizhu_server.snapshot_files import copy_snapshot_file
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

DEFAULT_TASK = ROOT / ".artifacts/china-landscape-from-scratch-20260924"
DEFAULT_DATA = ROOT / ".cache/training/china-landscape-public"
POLICY = "landscape-species-raw-public-v1"


def prepare(root, task):
    require_dataset_active(root)
    digest = sha256(task / "targets.json")
    destination = root / "species.json"
    if destination.exists():
        document = json.loads(destination.read_text())
        if document.get("policy") != POLICY or document.get("targetManifestSha256") != digest:
            raise ValueError("Target scope changed; use a new collection directory.")
        return document
    target = json.loads((task / "targets.json").read_text())
    species = target["classes"]
    if (
        target["level"] != "species" or not species
        or len({r["speciesId"] for r in species}) != len(species)
        or len({r["scientificName"] for r in species}) != len(species)
        or any(not r.get("chineseName") for r in species)
    ):
        raise ValueError("Require distinct, Chinese-named, approved species.")
    document = {
        "createdAt": now(), "policy": POLICY, "targetManifestSha256": digest,
        "species": species, "splitTargets": TARGETS,
        "unresolvedInputCount": target["unresolvedInputCount"],
        "reservedSourceNames": {r["scientificName"]: r["speciesId"] for r in species},
        "note": "Collection targets are not trained support; no weights/features reused.",
    }
    policy_path = task / "collection-policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        deferred = policy["deferredSpeciesIds"]
        if (
            policy["targetManifestSha256"] != digest
            or len(set(deferred)) != len(deferred)
            or not set(deferred) <= {r["speciesId"] for r in species}
        ):
            raise ValueError("Deferred collection scope differs from approved targets.")
        document["deferredSpeciesIds"] = deferred
        document["collectionPolicySha256"] = sha256(policy_path)
    write_json(destination, document)
    return document


def seed(root, task, document, *, reflink=False):
    """Import the audited union without changing original files or observer splits."""
    marker = root / "seed-import.json"
    inventory_path = task / "verified-image-files.json"
    identity = {
        "inventorySha256": sha256(inventory_path),
        "targetSpeciesSha256": sha256(root / "species.json"),
    }
    if marker.exists():
        report = json.loads(marker.read_text())
        if any(report.get(k) != v for k, v in identity.items()):
            raise ValueError("Seed inputs changed.")
        return report
    if (root / "collection.json").exists():
        raise ValueError("Cannot seed after collection has started.")
    inventory = json.loads(inventory_path.read_text())["files"]
    sources = json.loads((task / "verified-image-inventory.json").read_text())[
        "scannedPublicDatasets"
    ]
    selected_ids = {r["speciesId"] for r in inventory}
    lookup, mappings = {}, {}
    for relative in sources:
        source = ROOT.parent / relative
        require_dataset_active(source)
        for row in load_samples(source):
            lookup[(str(source / row["path"]), row["imageSha256"])] = row
        for filename in (source / "taxon-matches").glob("*.json"):
            record = json.loads(filename.read_text())
            if record["status"] != "matched" or record["speciesId"] not in selected_ids:
                continue
            identifier = record["speciesId"]
            if identifier in mappings and (
                mappings[identifier]["inatTaxonId"] != record["inatTaxonId"]
            ):
                raise ValueError("Source taxon mappings conflict.")
            mappings[identifier] = record
    # Some earlier raw-image exports omitted the source taxon mapping. Accept only
    # an explicit freshly resolved task record, never guess an ID from a filename.
    for filename in (task / "taxon-matches").glob("*.json"):
        record = json.loads(filename.read_text())
        identifier = record["speciesId"]
        if record["status"] != "matched" or identifier not in selected_ids:
            continue
        if identifier in mappings and mappings[identifier]["inatTaxonId"] != record["inatTaxonId"]:
            raise ValueError("Task and source taxon mappings conflict.")
        mappings[identifier] = record
    rows, files = [], []
    for item in inventory:
        source = ROOT.parent / item["path"]
        row = lookup[(str(source), item["sha256"])]
        if row["speciesId"] != item["speciesId"] or row["split"] != item["split"]:
            raise ValueError("Audited sample identity changed.")
        rows.append(row)
        files.append(source)
    validate_samples(rows, document["species"])
    wanted = {r["speciesId"]: r for r in document["species"]}
    selected = {r["speciesId"] for r in rows}
    if not selected <= mappings.keys():
        raise ValueError("Seed images need explicit source taxon mapping evidence.")
    selected_mappings = {identifier: mappings[identifier] for identifier in selected}
    if len({r["inatTaxonId"] for r in selected_mappings.values()}) != len(selected):
        raise ValueError("Multiple labels map to the same source taxon.")
    for identifier, record in selected_mappings.items():
        if record["scientificName"] != wanted[identifier]["scientificName"]:
            raise ValueError("Source scientific name changed.")
    grouped = defaultdict(list)
    for row, source in zip(rows, files, strict=True):
        if not source.resolve().is_relative_to((ROOT / ".cache/training").resolve()):
            raise ValueError("Only public training-pool images may be reused.")
        if sha256(source) != row["imageSha256"]:
            raise ValueError("Source image changed since audit.")
        destination = root / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if shutil.disk_usage(root).free < 4 * 1024**3:
                raise OSError("Keep at least 4 GiB free while seeding.")
            copy_snapshot_file(source, destination, reflink=reflink)
        if sha256(destination) != row["imageSha256"]:
            raise ValueError("Imported image checksum mismatch.")
        grouped[row["speciesId"]].append(row)
    for identifier, samples in grouped.items():
        record = selected_mappings[identifier]
        filename = hashlib.sha256(identifier.encode()).hexdigest() + ".json"
        write_json(root / "taxon-matches" / filename, record)
        counts = Counter(r["split"] for r in samples)
        write_json(root / "samples" / f"{record['inatTaxonId']}.json", {
            "samples": samples, "counts": dict(counts),
            "complete": all(counts[k] >= n for k, n in TARGETS.items()),
        })
    report = {
        **identity, "createdAt": now(), "copiedImages": len(rows),
        "copiedSpecies": len(grouped), "sourcePreserved": True,
        "copyMethod": "reflink-independent-inodes" if reflink else "copy",
    }
    write_json(marker, report)
    return report


def schedule(document, rows):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["speciesId"]][row["split"]] += 1
    missing = [
        species for species in document["species"]
        if any(counts[species["speciesId"]][k] < n for k, n in TARGETS.items())
    ]
    # Osmanthus first; complete near-ready species next. Membership never changes.
    missing.sort(key=lambda r: (
        r["scientificName"] != "Osmanthus fragrans",
        -sum(min(counts[r["speciesId"]][k], n) for k, n in TARGETS.items()),
        r["scientificName"],
    ))
    return {**document, "species": missing}


def collection_schedule(document, rows):
    """Defer previously exhausted source attempts without hiding target shortfalls."""
    missing = schedule(document, rows)
    deferred = set(document.get("deferredSpeciesIds", []))
    return {**missing, "species": [r for r in missing["species"] if r["speciesId"] not in deferred]}


def overall_coverage(root, document):
    rows = load_samples(root)
    validate_samples(rows, document["species"])
    missing = schedule(document, rows)["species"]
    missing_ids = {r["speciesId"] for r in missing}
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["speciesId"]][row["split"]] += 1
    result = {
        "updatedAt": now(), "approvedSpecies": len(document["species"]),
        "images": len(rows), "coveredSpecies": len(counts),
        "readySpeciesCount": len(document["species"]) - len(missing),
        "readySpeciesIds": [r["speciesId"] for r in document["species"]
                            if r["speciesId"] not in missing_ids],
        "shortfalls": [{
            "speciesId": r["speciesId"], "scientificName": r["scientificName"],
            "counts": dict(counts[r["speciesId"]]),
            "missing": {k: max(0, v - counts[r["speciesId"]][k]) for k, v in TARGETS.items()},
            "deferredFromThisSourcePass": r["speciesId"] in document.get("deferredSpeciesIds", []),
        } for r in missing],
        "unresolvedInputCount": document["unresolvedInputCount"],
        "trainedSupportUpdated": False,
    }
    write_json(root / "overall-coverage.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--reflink", action="store_true",
                        help="Independent copy-on-write seed; fails rather than falling back")
    args = parser.parse_args()
    with dataset_lock(args.data):
        document = prepare(args.data, args.task)
        report = seed(args.data, args.task, document, reflink=args.reflink)
        scheduled = collection_schedule(document, load_samples(args.data))
        coverage = overall_coverage(args.data, document)
        write_json(args.data / "run-order.json", {
            "updatedAt": now(), "targetSpeciesSha256": sha256(args.data / "species.json"),
            "approvedSpeciesCount": len(document["species"]),
            "collectionSpeciesCount": len(scheduled["species"]),
            "speciesIds": [r["speciesId"] for r in scheduled["species"]],
            "seedImages": report["copiedImages"],
            "deferredShortfallSpecies": len(coverage["shortfalls"]) - len(scheduled["species"]),
        })
        print(f"Seeded {report['copiedImages']} photos; "
              f"{len(scheduled['species'])} species need collection.", flush=True)
        if args.action == "collect":
            client = PublicClient()
            try:
                collect(
                    args.data, client, scheduled,
                    class_limit=len(scheduled["species"]), split_targets=TARGETS,
                    max_pages=20, resolve_species=TaxonResolver(
                        args.data, client, allow_source_names=True,
                        target_names=document["reservedSourceNames"],
                    ),
                )
                overall_coverage(args.data, document)
            except Exception as exc:
                write_json(args.data / "stopped.json", {
                    "updatedAt": now(), "errorType": type(exc).__name__,
                    "automaticRetry": False,
                })
                raise


if __name__ == "__main__":
    main()
