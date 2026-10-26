"""Append a failed one-shot independent set as balanced training replay features."""

import argparse
import json
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def create_plan(features, manifest_path, results_path, output, node_root):
    if output.exists():
        raise ValueError("Use a new replay feature plan.")
    index = json.loads((features / "index.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        index.get("status") != "features-extracted"
        or sha256(features / "features.safetensors") != index.get("featuresSha256")
        or manifest.get("status") != "complete"
        or manifest.get("imagesVerified") is not True
        or manifest.get("modelOutputsReadDuringSelection") is not False
        or results.get("status") != "final-independent-evaluation"
        or results.get("goalMet") is not False
        or results.get("manifestSha256") != sha256(manifest_path)
        or results.get("representedSpecies") != manifest.get("representedSpecies")
    ):
        raise ValueError("Require one complete failed independent evaluation.")
    rows = list(index["samples"])
    ids = {row["sampleId"] for row in rows}
    hashes = {row["imageSha256"] for row in rows}
    replay = []
    for sample in manifest["samples"]:
        raw = Path(sample["path"])
        image = raw if raw.is_absolute() else (manifest_path.parent / raw).resolve()
        if not image.is_file() or sha256(image) != sample["imageSha256"]:
            raise ValueError("Failed-independent replay image changed.")
        sample_id = f"replay-failed:{sample['sampleId']}"
        if sample_id in ids or sample["imageSha256"] in hashes:
            raise ValueError("Failed-independent replay duplicates prior features.")
        row = {
            "sampleId": sample_id, "speciesId": sample["speciesId"],
            "imageSha256": sample["imageSha256"],
            "observerGroup": sample["observerGroup"],
            "photoId": f"replay-photo:{sample['sourceRecordId']}",
            "observationId": f"replay-observation:{sample['sourceRecordId']}",
            "split": "train",
        }
        replay.append({
            **row, "repositoryPath": str(image),
            "sourcePath": str((node_root / image.relative_to(Path.cwd())).resolve()),
        })
        rows.append(row)
        ids.add(sample_id)
        hashes.add(sample["imageSha256"])
    rows.sort(key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])))
    replay.sort(key=lambda row: (row["speciesId"], row["sampleId"]))
    output.mkdir(parents=True)
    partition = output / "partition-0.json"
    write_json(partition, {
        "createdAt": now(), "purpose": "target-species-bioclip-feature-partition",
        "node": str(node_root), "samples": replay,
    })
    (output / "files-0.txt").write_text("".join(row["repositoryPath"] + "\n" for row in replay))
    write_json(output / "plan.json", {
        "createdAt": now(), "status": "failed-independent-replay-feature-plan",
        "architecture": index["architecture"],
        "checkpointSha256": index["checkpointSha256"],
        "dimension": index["dimension"], "normalized": index["normalized"],
        "samples": rows, "baseFeatureRows": len(index["samples"]),
        "replayedFailedIndependentSamples": len(replay),
        "finalFeatureRows": len(rows), "reusedFeatureRoot": str(features),
        "reusedFeatureIndexSha256": sha256(features / "index.json"),
        "reusedFeatureWeightsSha256": sha256(features / "features.safetensors"),
        "failedIndependentManifestSha256": sha256(manifest_path),
        "failedIndependentResultsSha256": sha256(results_path),
        "failedIndependentGoalMet": False,
        "partitions": [{"partition": str(partition), "nodeRoot": str(node_root),
                        "samples": len(replay)}],
        "productionChanged": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    args = parser.parse_args()
    create_plan(args.features.resolve(), args.manifest.resolve(), args.results.resolve(),
                args.output.resolve(), args.node_root.resolve())


if __name__ == "__main__":
    main()
