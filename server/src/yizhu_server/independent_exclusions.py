"""Build immutable exclusions from a snapshot and all prior independent manifests."""

import argparse
import json
import re
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def build(snapshot, manifests):
    training_path = snapshot / "train-validation.json"
    test_path = snapshot / "test-heldout.json"
    training = json.loads(training_path.read_text())
    test = json.loads(test_path.read_text())
    if (
        training.get("testImagesIncluded") is not False
        or test.get("trainingManifestSha256") != sha256(training_path)
        or test.get("classes") != training.get("classes")
    ):
        raise ValueError("Snapshot manifests differ.")
    occurrences, urls, photos, observations, hashes = set(), set(), set(), set(), set()

    def add_row(row):
        hashes.add(row["imageSha256"])
        image_url = row.get("imageUrl") or row.get("sourceImageUrl")
        if image_url:
            urls.add(image_url)
            match = re.search(r"/photos/(\d+)/(?:original|medium|large|small|square)", image_url)
            if match:
                photos.add(match.group(1))
        observation = str(row.get("observationId", ""))
        if observation.startswith("gbif:"):
            occurrences.add(observation.split(":", 1)[1])
        elif observation.isdigit():
            observations.add(observation)
        source_record = row.get("sourceRecordId")
        if source_record is not None:
            occurrences.add(str(source_record))

    for row in training["samples"] + test["samples"]:
        add_row(row)
    evidence = []
    for path in manifests:
        document = json.loads(path.read_text())
        if (
            document.get("purpose")
            != "new-independent-test-never-used-for-training-or-selection"
            or document.get("modelOutputsReadDuringSelection") is not False
            or document.get("representedSpecies")
            != len({row["speciesId"] for row in document.get("samples", [])})
        ):
            raise ValueError("Prior independent manifest is invalid.")
        for row in document["samples"]:
            add_row(row)
        evidence.append({
            "path": str(path), "sha256": sha256(path),
            "samples": len(document["samples"]),
        })
    return {
        "createdAt": now(),
        "purpose": "exclude-training-development-and-all-prior-independent-images",
        "gbifOccurrenceKeys": sorted(occurrences),
        "sourceImageUrls": sorted(urls),
        "inatPhotoIds": sorted(photos),
        "inatObservationIds": sorted(observations),
        "imageSha256": sorted(hashes),
        "snapshotTrainingSha256": sha256(training_path),
        "snapshotTestSha256": sha256(test_path),
        "priorIndependentManifests": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable exclusion file.")
    result = build(
        args.snapshot.resolve(), [path.resolve() for path in args.exclude_manifest],
    )
    write_json(args.output.resolve(), result)
    print(json.dumps({
        "occurrences": len(result["gbifOccurrenceKeys"]),
        "urls": len(result["sourceImageUrls"]),
        "hashes": len(result["imageSha256"]),
        "priorManifests": len(result["priorIndependentManifests"]),
    }))


if __name__ == "__main__":
    main()
