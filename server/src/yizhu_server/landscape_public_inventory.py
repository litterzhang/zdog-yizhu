"""Audit reusable licensed images for a versioned target set; never read model features."""

import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from yizhu_server.everyday_training_data import TARGETS
from yizhu_server.training_classifier import validate_samples
from yizhu_server.training_data import (
    ROOT,
    DuplicateIndex,
    dataset_lock,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)


def select_reusable(candidates, classes):
    """Preserve existing split identities; conflicting reuse is an error, not a relabel."""
    wanted = {r["speciesId"]: r["scientificName"] for r in classes}
    conflicts = {}
    for row, _ in candidates:
        if row["speciesId"] not in wanted:
            continue
        identity = (row["speciesId"], row["scientificName"], row["split"], row["observerGroup"])
        for key in (("hash", row["imageSha256"]), ("sample", row["sampleId"]),
                    ("photo", row["photoId"]), ("observation", row["observationId"])):
            if key in conflicts and conflicts[key] != identity:
                raise ValueError("A reused image has conflicting class/split/observer identity.")
            conflicts[key] = identity
    selected, rejected = [], Counter()
    counts, observers = defaultdict(Counter), Counter()
    duplicates = DuplicateIndex()
    ids, photos, observations = set(), set(), set()
    for row, path in candidates:
        sid = row["speciesId"]
        if sid not in wanted:
            continue
        # Validate each row even if it would be removed as a duplicate/quota surplus.
        validate_samples([row], classes)
        perceptual = int(row["differenceHash"], 16)
        if (
            row["sampleId"] in ids or row["photoId"] in photos
            or row["observationId"] in observations
            or duplicates.contains(row["imageSha256"], perceptual)
        ):
            rejected["duplicateOrNearDuplicate"] += 1
            continue
        if counts[sid][row["split"]] >= TARGETS[row["split"]]:
            rejected["splitQuotaAlreadyMet"] += 1
            continue
        if observers[(sid, row["observerGroup"])] >= 3:
            rejected["observerCap"] += 1
            continue
        duplicates.add(row["imageSha256"], perceptual)
        ids.add(row["sampleId"])
        photos.add(row["photoId"])
        observations.add(row["observationId"])
        counts[sid][row["split"]] += 1
        observers[(sid, row["observerGroup"])] += 1
        selected.append((row, path))
    validate_samples([r for r, _ in selected], classes)
    return selected, rejected


def audit(target_path, sources, output, *, completed_source_pool=None):
    from PIL import Image

    if output.exists():
        raise ValueError("Use a new inventory directory; do not overwrite frozen evidence.")
    source_paths = [path.resolve() for path in sources]
    if len(set(source_paths)) != len(source_paths) or not source_paths:
        raise ValueError("Require distinct public source directories.")
    allowed_root = (ROOT / ".cache/training").resolve()
    target = json.loads(target_path.read_text())
    classes = target["classes"]
    if target["level"] != "species" or len({r["speciesId"] for r in classes}) != len(classes):
        raise ValueError("Require a distinct species-level target set.")
    with ExitStack() as stack:
        for source in sorted(source_paths):
            if not source.is_relative_to(allowed_root):
                raise ValueError("Source is outside public training pools.")
            require_dataset_active(source)
            stack.enter_context(dataset_lock(source))
        candidates = [
            (row, source / row["path"]) for source in source_paths for row in load_samples(source)
        ]
        selected, rejected = select_reusable(candidates, classes)
        files, counts = [], defaultdict(Counter)
        for row, path in selected:
            if not path.resolve().is_relative_to(allowed_root) or not path.is_file():
                raise ValueError("Public image missing or escapes public storage.")
            if sha256(path) != row["imageSha256"]:
                raise ValueError("Public image checksum differs from source metadata.")
            with Image.open(path) as image:
                image.verify()
            files.append({
                "speciesId": row["speciesId"], "split": row["split"],
                "sha256": row["imageSha256"], "path": str(path.relative_to(ROOT.parent)),
                "license": row["license"],
            })
            counts[row["speciesId"]][row["split"]] += 1
        ready = [
            r["speciesId"] for r in classes
            if all(counts[r["speciesId"]][k] >= v for k, v in TARGETS.items())
        ]
        deferred, source_progress = [], None
        if completed_source_pool:
            pool = completed_source_pool.resolve()
            if pool not in source_paths:
                raise ValueError("Completed source pool must be among the locked sources.")
            source_progress = json.loads((pool / "progress.json").read_text())
            coverage = json.loads((pool / "coverage.json").read_text())
            if source_progress["stage"] != "collection-pass-finished":
                raise ValueError("Cannot defer an unfinished collection pass.")
            previous = json.loads((pool / "species.json").read_text())["species"]
            previous_ids = {r["speciesId"] for r in previous}
            covered_ids = set(coverage["readySpecies"]) | {
                r["speciesId"] for r in coverage["shortfalls"]
            }
            if not covered_ids <= previous_ids or len(covered_ids) != coverage["targetSpecies"]:
                raise ValueError("Previous collection scope cannot be verified.")
            deferred = [
                r["speciesId"] for r in classes
                if r["speciesId"] in covered_ids and r["speciesId"] not in ready
            ]
        output.mkdir(parents=True)
        # Copy bytes, rather than rewriting the target version/checksum.
        (output / "targets.json").write_bytes(target_path.read_bytes())
        report = {
            "checkedAt": now(), "resolvedTargetSpecies": len(classes),
            "targetManifestSha256": sha256(target_path),
            "scannedPublicDatasets": [str(p.relative_to(ROOT.parent)) for p in source_paths],
            "validDistinctImages": len(files),
            "coveredSpecies": sum(bool(counts[r["speciesId"]]) for r in classes),
            "readySpeciesCount": len(ready), "readySpeciesIds": ready,
            "rejectedMetadataRows": dict(rejected),
            "perSpecies": [
                {**{k: r[k] for k in ("speciesId", "scientificName", "chineseName")},
                 "counts": dict(counts[r["speciesId"]]),
                 "total": sum(counts[r["speciesId"]].values())}
                for r in classes
            ],
            "rawImagesOnly": True, "modelWeightsOrFeaturesRead": False,
            "splitAssignmentsUnchanged": True,
        }
        write_json(output / "verified-image-files.json", {"checkedAt": now(), "files": files})
        write_json(output / "verified-image-inventory.json", report)
        write_json(output / "collection-policy.json", {
            "createdAt": now(), "targetManifestSha256": sha256(target_path),
            "deferredSpeciesIds": deferred,
            "reason": "Already attempted in the completed iNaturalist pass; need another source "
                      "or an explicitly approved later pass, not immediate repeated requests.",
            "previousPool": str(completed_source_pool) if completed_source_pool else None,
            "previousProgress": source_progress,
            "deferredIsNotReady": True, "allTargetsPreserved": True,
        })
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-source-pool", type=Path)
    args = parser.parse_args()
    report = audit(args.targets, args.sources, args.output,
                   completed_source_pool=args.completed_source_pool)
    print(json.dumps({k: report[k] for k in (
        "resolvedTargetSpecies", "validDistinctImages", "coveredSpecies", "readySpeciesCount",
    )}))


if __name__ == "__main__":
    main()
