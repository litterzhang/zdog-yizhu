"""Build a reference-only target snapshot from an audited inventory and a prior snapshot."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from yizhu_server.training_data import (
    ROOT,
    DuplicateIndex,
    load_samples,
    now,
    sha256,
    write_json,
)


def _perceptual(row):
    return int(row["differenceHash"], 16)


def merge_rows(primary, supplements, classes):
    """Keep the audited primary pool, then add non-conflicting supplement rows."""
    class_by_id = {row["speciesId"]: row for row in classes}
    if len(class_by_id) != len(classes):
        raise ValueError("Target species IDs must be unique.")
    selected, rejected = [], Counter()
    duplicates = DuplicateIndex()
    identities = {}

    def identity(row):
        return row["speciesId"], row["split"], row["observerGroup"]

    for pool_name, rows in [("primary", primary), *supplements]:
        for row in rows:
            target = class_by_id.get(row["speciesId"])
            if target is None:
                rejected[f"{pool_name}:outside-target"] += 1
                continue
            if row["scientificName"] != target["scientificName"]:
                raise ValueError("A sample scientific name differs from the target species.")
            current = identity(row)
            keys = [
                ("sample", str(row["sampleId"])),
                ("photo", str(row["photoId"])),
                ("observation", str(row["observationId"])),
                ("sha256", row["imageSha256"]),
            ]
            for key in keys:
                previous = identities.get(key)
                if previous is not None and previous != current:
                    raise ValueError(f"Conflicting sample identity: {key[0]}.")
            perceptual = _perceptual(row)
            if duplicates.contains(row["imageSha256"], perceptual):
                rejected[f"{pool_name}:exact-or-near-duplicate"] += 1
                continue
            for key in keys:
                identities[key] = current
            duplicates.add(row["imageSha256"], perceptual)
            selected.append(row)
    return selected, rejected


def _inventory_rows(inventory_root, source_roots):
    verified_path = inventory_root / "verified-image-files.json"
    verified = json.loads(verified_path.read_text())["files"]
    by_path = {}
    for source in source_roots:
        for row in load_samples(source):
            path = (source / row["path"]).resolve()
            if path in by_path:
                raise ValueError("A source image path occurs in multiple source manifests.")
            by_path[path] = row
    rows = []
    for item in verified:
        path = (ROOT.parent / item["path"]).resolve()
        row = by_path.get(path)
        if (
            row is None
            or row["speciesId"] != item["speciesId"]
            or row["split"] != item["split"]
            or row["imageSha256"] != item["sha256"]
            or row["license"] != item["license"]
            or not path.is_file()
            or sha256(path) != item["sha256"]
        ):
            raise ValueError("Verified inventory no longer matches its source manifest.")
        rows.append({
            **row,
            "sourceDataset": str(path.parent.parent.relative_to(ROOT.parent)),
            "sourcePath": str(path.relative_to(ROOT.parent)),
        })
    return rows, verified_path


def _snapshot_rows(snapshot):
    training_path = snapshot / "train-validation.json"
    test_path = snapshot / "test-heldout.json"
    training = json.loads(training_path.read_text())
    test = json.loads(test_path.read_text())
    if (
        training.get("testImagesIncluded") is not False
        or test.get("classes") != training.get("classes")
        or test.get("trainingManifestSha256") != sha256(training_path)
    ):
        raise ValueError("Supplement snapshot manifests differ.")
    rows = []
    for row in training["samples"] + test["samples"]:
        path = (snapshot / row["path"]).resolve()
        if not path.is_file() or sha256(path) != row["imageSha256"]:
            raise ValueError("Supplement snapshot image changed.")
        rows.append({
            **row,
            "sourceDataset": str(snapshot.relative_to(ROOT.parent)),
            "sourcePath": str(path.relative_to(ROOT.parent)),
        })
    return rows, training["classes"], training_path, test_path


def build(target_path, inventory, sources, supplements, output):
    if output.exists():
        raise ValueError("Use a new target snapshot directory.")
    target = json.loads(target_path.read_text())
    classes = target["classes"]
    if target.get("level") != "species" or len(classes) != 249:
        raise ValueError("Require the reviewed 249-species target list.")
    primary, inventory_path = _inventory_rows(inventory, sources)
    supplement_rows = []
    source_evidence = []
    for index, snapshot in enumerate(supplements, 1):
        rows, snapshot_classes, training_path, test_path = _snapshot_rows(snapshot)
        if not {row["speciesId"] for row in snapshot_classes} <= {
            row["speciesId"] for row in classes
        }:
            raise ValueError("Supplement contains species outside the target list.")
        name = f"supplement-{index}"
        supplement_rows.append((name, rows))
        source_evidence.append({
            "path": str(snapshot),
            "trainingManifestSha256": sha256(training_path),
            "testManifestSha256": sha256(test_path),
            "rows": len(rows),
        })
    merged, rejected = merge_rows(primary, supplement_rows, classes)
    counts = defaultdict(Counter)
    for row in merged:
        counts[row["speciesId"]][row["split"]] += 1
    no_train = [
        {**row, "counts": dict(counts[row["speciesId"]])}
        for row in classes if counts[row["speciesId"]]["train"] == 0
    ]
    covered = sum(bool(counts[row["speciesId"]]) for row in classes)
    output.mkdir(parents=True)
    write_json(output / "species.json", {
        "createdAt": now(), "policy": "reviewed-target-reference-union-v1",
        "species": classes, "requestedClassCount": len(classes),
        "targetManifestSha256": sha256(target_path),
    })
    train_validation = sorted(
        (row for row in merged if row["split"] != "test"),
        key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])),
    )
    test = sorted(
        (row for row in merged if row["split"] == "test"),
        key=lambda row: (row["speciesId"], str(row["sampleId"])),
    )
    write_json(output / "train-validation.json", {
        "createdAt": now(), "classes": classes,
        "targetSpeciesIds": [row["speciesId"] for row in classes],
        "samples": train_validation, "testImagesIncluded": False,
        "pretrainedWeightsAllowed": True, "pretrainedClassifierRowsAllowed": True,
        "automaticDeployment": False,
        "scope": "249 reviewed species; reference-only union, source images are not copied.",
    })
    write_json(output / "test-heldout.json", {
        "createdAt": now(), "classes": classes, "samples": test,
        "trainingManifestSha256": sha256(output / "train-validation.json"),
        "independentFinalBenchmark": False,
        "note": "Development holdout inherited without resplitting; not final acceptance data.",
    })
    write_json(output / "audit.json", {
        "createdAt": now(), "status": "reference-snapshot-audited",
        "requestedClassCount": len(classes), "coveredClassCount": covered,
        "trainingClassCount": len(classes) - len(no_train),
        "missingTrainingClassCount": len(no_train),
        "missingTrainingClasses": no_train,
        "primaryRows": len(primary),
        "supplementRows": sum(len(rows) for _, rows in supplement_rows),
        "mergedDistinctRows": len(merged),
        "splitCounts": dict(Counter(row["split"] for row in merged)),
        "perSpecies": [{
            **{key: row[key] for key in ("speciesId", "scientificName", "chineseName")},
            "counts": dict(counts[row["speciesId"]]),
        } for row in classes],
        "rejected": dict(rejected),
        "targetManifestSha256": sha256(target_path),
        "inventoryManifestSha256": sha256(inventory_path),
        "supplements": source_evidence,
        "imagesCopied": False, "splitsChanged": False,
        "modelWeightsOrFeaturesRead": False, "productionChanged": False,
    })
    return json.loads((output / "audit.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.targets.resolve(), args.inventory.resolve(),
        [path.resolve() for path in args.source],
        [path.resolve() for path in args.supplement], args.output.resolve(),
    )
    print(json.dumps({
        key: report[key] for key in (
            "requestedClassCount", "coveredClassCount", "trainingClassCount",
            "missingTrainingClassCount", "mergedDistinctRows", "splitCounts",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
