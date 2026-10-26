"""Plan and merge incremental BioCLIP features for an expanded reviewed target set."""

import argparse
import json
import os
from pathlib import Path

from yizhu_server.training_data import ROOT, now, sha256, write_json

IDENTITY_KEYS = (
    "sampleId", "speciesId", "imageSha256", "observerGroup",
    "photoId", "observationId", "split",
)


def _identity(row):
    return tuple(str(row[key]) for key in IDENTITY_KEYS)


def create_plan(snapshot, reused, output, node_roots):
    if output.exists() or len(node_roots) < 1:
        raise ValueError("Use a new plan directory and at least one feature node.")
    training_path = snapshot / "train-validation.json"
    test_path = snapshot / "test-heldout.json"
    training = json.loads(training_path.read_text())
    test = json.loads(test_path.read_text())
    classes = training["classes"]
    if (
        training.get("testImagesIncluded") is not False
        or test.get("classes") != classes
        or test.get("trainingManifestSha256") != sha256(training_path)
    ):
        raise ValueError("Expanded target manifests differ.")
    target_rows = training["samples"] + test["samples"]
    target_ids = {row["speciesId"] for row in classes}
    existing = json.loads((reused / "index.json").read_text())
    if (
        existing.get("status") != "features-extracted"
        or existing.get("architecture") != "imageomics/bioclip-2"
        or existing.get("dimension") != 768
        or sha256(reused / "features.safetensors") != existing.get("featuresSha256")
    ):
        raise ValueError("Reusable BioCLIP feature source changed.")
    reused_by_id = {row["sampleId"]: row for row in existing["samples"]}
    if len(reused_by_id) != len(existing["samples"]):
        raise ValueError("Reusable feature sample IDs are not unique.")
    desired = {row["sampleId"]: row for row in target_rows}
    if len(desired) != len(target_rows):
        raise ValueError("Expanded target sample IDs are not unique.")

    replayed = []
    for row in existing["samples"]:
        if row["sampleId"] in desired:
            continue
        if (
            row["speciesId"] in target_ids
            and row["split"] == "train"
            and str(row["sampleId"]).startswith(("replay1:", "replay2:"))
        ):
            replayed.append(row)
            desired[row["sampleId"]] = row

    reused_ids, missing = [], []
    for row in target_rows:
        prior = reused_by_id.get(row["sampleId"])
        if prior is not None:
            if _identity(prior) != _identity(row):
                raise ValueError("A reusable feature has changed sample identity.")
            reused_ids.append(row["sampleId"])
            continue
        source = (ROOT.parent / row["sourcePath"]).resolve()
        if (
            not source.is_relative_to(ROOT.parent.resolve())
            or not source.is_file()
            or sha256(source) != row["imageSha256"]
        ):
            raise ValueError("An expanded-target image is unavailable or changed.")
        missing.append((row, source, source.stat().st_size))

    final_rows = sorted(
        desired.values(),
        key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])),
    )
    output.mkdir(parents=True)
    buckets = [[] for _ in node_roots]
    sizes = [0 for _ in node_roots]
    for row, source, size in sorted(missing, key=lambda value: value[2], reverse=True):
        index = min(range(len(sizes)), key=sizes.__getitem__)
        repository_path = str(source.relative_to(ROOT.parent))
        buckets[index].append({
            **{key: row[key] for key in IDENTITY_KEYS},
            "repositoryPath": repository_path,
            "sourcePath": str(node_roots[index] / repository_path),
        })
        sizes[index] += size
    partitions = []
    for index, rows in enumerate(buckets):
        rows.sort(key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])))
        path = output / f"partition-{index}.json"
        write_json(path, {
            "createdAt": now(), "purpose": "target-species-bioclip-feature-partition",
            "node": str(node_roots[index]), "samples": rows,
        })
        (output / f"files-{index}.txt").write_text(
            "".join(row["repositoryPath"] + "\n" for row in rows)
        )
        partitions.append({
            "partition": str(path), "nodeRoot": str(node_roots[index]),
            "samples": len(rows), "sourceBytes": sizes[index],
        })
    write_json(output / "plan.json", {
        "createdAt": now(), "status": "incremental-feature-plan",
        "architecture": existing["architecture"],
        "checkpointSha256": existing["checkpointSha256"],
        "dimension": existing["dimension"], "normalized": existing["normalized"],
        "requestedClassCount": len(classes), "targetSnapshotRows": len(target_rows),
        "reusedTargetRows": len(reused_ids), "replayedTrainingRows": len(replayed),
        "newFeatureRows": len(missing), "finalFeatureRows": len(final_rows),
        "classes": classes, "samples": final_rows,
        "reusedFeatureRoot": str(reused),
        "reusedFeatureIndexSha256": sha256(reused / "index.json"),
        "reusedFeatureWeightsSha256": sha256(reused / "features.safetensors"),
        "trainingManifestSha256": sha256(training_path),
        "testManifestSha256": sha256(test_path),
        "partitions": partitions, "productionChanged": False,
    })
    return json.loads((output / "plan.json").read_text())


def merge(plan_path, reused, partitions, output):
    import torch
    from safetensors.torch import load_file, save_file

    if output.exists():
        raise ValueError("Use a new merged feature directory.")
    plan = json.loads(plan_path.read_text())
    base_index = json.loads((reused / "index.json").read_text())
    base_vectors = load_file(reused / "features.safetensors")["features"]
    if (
        sha256(reused / "index.json") != plan["reusedFeatureIndexSha256"]
        or sha256(reused / "features.safetensors") != plan["reusedFeatureWeightsSha256"]
        or base_vectors.shape != (len(base_index["samples"]), plan["dimension"])
    ):
        raise ValueError("Reusable feature source differs from the frozen plan.")
    wanted = {row["sampleId"]: row for row in plan["samples"]}
    gathered = {}
    for index, row in enumerate(base_index["samples"]):
        expected = wanted.get(row["sampleId"])
        if expected is not None:
            if _identity(row) != _identity(expected):
                raise ValueError("Reusable sample identity differs from the plan.")
            gathered[row["sampleId"]] = base_vectors[index]
    for root in partitions:
        index = json.loads((root / "index.json").read_text())
        vectors = load_file(root / "features.safetensors")["features"]
        if (
            index.get("architecture") != plan["architecture"]
            or index.get("checkpointSha256") != plan["checkpointSha256"]
            or index.get("dimension") != plan["dimension"]
            or index.get("normalized") != plan["normalized"]
            or sha256(root / "features.safetensors") != index.get("featuresSha256")
            or vectors.shape != (len(index["samples"]), plan["dimension"])
        ):
            raise ValueError("Incremental feature partition identity differs.")
        for position, row in enumerate(index["samples"]):
            expected = wanted.get(row["sampleId"])
            if (
                expected is None or row["sampleId"] in gathered
                or _identity(row) != _identity(expected)
            ):
                raise ValueError("Incremental feature sample changed or duplicated.")
            gathered[row["sampleId"]] = vectors[position]
    if set(gathered) != set(wanted):
        raise ValueError("Feature sources do not cover the complete expansion plan.")
    rows = plan["samples"]
    vectors = torch.stack([gathered[row["sampleId"]] for row in rows])
    output.mkdir(parents=True)
    temporary = output / "features.safetensors.tmp"
    save_file({"features": vectors.contiguous()}, temporary)
    os.replace(temporary, output / "features.safetensors")
    write_json(output / "index.json", {
        "createdAt": now(), "status": "features-extracted",
        "architecture": plan["architecture"],
        "checkpointSha256": plan["checkpointSha256"],
        "dimension": plan["dimension"], "normalized": plan["normalized"],
        "samples": [{"row": index, **row} for index, row in enumerate(rows)],
        "featuresSha256": sha256(output / "features.safetensors"),
        "sourcePlanSha256": sha256(plan_path),
        "sourcePartitions": [{
            "path": str(root), "indexSha256": sha256(root / "index.json"),
        } for root in partitions],
        "trainingStarted": False, "testUsedForTraining": False,
        "productionChanged": False,
    })
    write_json(output / "progress.json", {
        "updatedAt": now(), "stage": "complete", "completed": len(rows),
        "total": len(rows), "trainingStarted": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    planner = subparsers.add_parser("plan")
    planner.add_argument("--snapshot", type=Path, required=True)
    planner.add_argument("--reused", type=Path, required=True)
    planner.add_argument("--output", type=Path, required=True)
    planner.add_argument("--node-root", type=Path, action="append", required=True)
    merger = subparsers.add_parser("merge")
    merger.add_argument("--plan", type=Path, required=True)
    merger.add_argument("--reused", type=Path, required=True)
    merger.add_argument("--partition", type=Path, action="append", required=True)
    merger.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        report = create_plan(
            args.snapshot.resolve(), args.reused.resolve(), args.output.resolve(),
            [path.resolve() for path in args.node_root],
        )
        print(json.dumps({
            key: report[key] for key in (
                "requestedClassCount", "targetSnapshotRows", "reusedTargetRows",
                "replayedTrainingRows", "newFeatureRows", "finalFeatureRows",
            )
        }))
    else:
        merge(
            args.plan.resolve(), args.reused.resolve(),
            [path.resolve() for path in args.partition], args.output.resolve(),
        )


if __name__ == "__main__":
    main()
