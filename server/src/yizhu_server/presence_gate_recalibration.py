"""Recalibrate a fixed plant-presence head using failed-independent replay positives."""

import argparse
import json
import math
from pathlib import Path

from yizhu_server.plant_presence import (
    LABELS,
    PlantPresenceClassifier,
    binary_metrics,
    cached_feature,
    select_threshold,
)
from yizhu_server.recognition import read_photo
from yizhu_server.training_data import now, sha256, write_json


def recalibrate(head_path, binary_root, replay_manifest, output, *, minimum_recall=.99,
                threads=4):
    if output.exists() or not .95 <= minimum_recall <= 1:
        raise ValueError("Use a new output and 95–100% replay recall target.")
    classifier = PlantPresenceClassifier(head_path, threads=threads)
    binary_manifest = binary_root / "samples.json"
    binary_document = json.loads(binary_manifest.read_text())
    rows = binary_document.get("samples", [])
    binary_digest = sha256(binary_manifest)
    if (
        binary_document.get("version") != 1 or not rows
        or binary_digest != classifier.report.get("datasetSha256")
    ):
        raise ValueError("Binary validation manifest differs from the fixed head.")
    binary_scores, binary_labels = [], []
    for row in rows:
        if row["split"] != "validation":
            continue
        result = classifier.predict_vector(cached_feature(classifier.encoder, binary_root, row))
        binary_scores.append(result["score"])
        binary_labels.append(LABELS.index(row["label"]))
    manifest = json.loads(replay_manifest.read_text())
    replay_scores = []
    for index, row in enumerate(manifest["samples"], 1):
        raw = Path(row["path"])
        image_path = raw if raw.is_absolute() else (replay_manifest.parent / raw).resolve()
        if not image_path.is_file() or sha256(image_path) != row["imageSha256"]:
            raise ValueError("Replay-positive image changed.")
        with read_photo(image_path) as image:
            replay_scores.append(classifier.predict(image)["score"])
        if index % 40 == 0 or index == len(manifest["samples"]):
            print(json.dumps({"completed": index, "total": len(manifest["samples"])}), flush=True)
    binary_threshold = select_threshold(binary_scores, binary_labels, .99)
    ordered = sorted(replay_scores, reverse=True)
    replay_threshold = ordered[math.ceil(len(ordered) * minimum_recall) - 1]
    threshold = min(classifier.report["threshold"], binary_threshold, replay_threshold)
    contract = {
        "createdAt": now(), "policy": "min-fixed-head-binary99-replay99-v1",
        "headSha256": classifier.version, "encoder": classifier.report["encoder"],
        "threshold": threshold, "originalThreshold": classifier.report["threshold"],
        "minimumBinaryValidationPlantRecall": .99,
        "minimumReplayPlantRecall": minimum_recall,
        "binaryDatasetSha256": binary_digest,
        "binaryValidation": binary_metrics(binary_scores, binary_labels, threshold),
        "replayPlantValidation": {
            "samples": len(replay_scores),
            "plantRecall": sum(score >= threshold for score in replay_scores) / len(replay_scores),
            "minimumScore": min(replay_scores), "maximumScore": max(replay_scores),
        },
        "replayManifestSha256": sha256(replay_manifest),
        "confidenceCalibrated": False, "testUsedForSelection": False,
        "productionApproved": False,
    }
    write_json(output, contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--binary-data", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=.99)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = recalibrate(
        args.head.resolve(), args.binary_data.resolve(), args.replay_manifest.resolve(),
        args.output.resolve(), minimum_recall=args.minimum_recall, threads=args.threads,
    )
    print(json.dumps({
        "threshold": result["threshold"],
        "binaryValidation": result["binaryValidation"],
        "replayPlantValidation": result["replayPlantValidation"],
    }, indent=2))


if __name__ == "__main__":
    main()
