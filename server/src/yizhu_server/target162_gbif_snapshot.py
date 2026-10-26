"""Build a reviewed 162-target snapshot by adding exact-taxonomy GBIF candidates."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from yizhu_server.snapshot_files import copy_snapshot_file
from yizhu_server.training_data import (
  DuplicateIndex,
  group_split,
  now,
  sha256,
  write_json,
)


def _perceptual(value):
  return int(value, 16) if isinstance(value, str) else int(value)


def _license(url):
  value = (url or "").lower()
  if "publicdomain/zero/1.0" in value:
    return "cc0"
  if "licenses/by/4.0" in value:
    return "cc-by"
  raise ValueError("GBIF candidate media license changed.")


def _author(candidate):
  return str(
    candidate.get("mediaCreator") or candidate.get("mediaRightsHolder")
    or candidate.get("recordedBy") or candidate.get("datasetKey")
  ).strip()


def build(base, collections, output, *, reflink=True):
  if output.exists():
    raise ValueError("Use a new snapshot directory.")
  training_path = base / "train-validation.json"
  test_path = base / "test-heldout.json"
  training = json.loads(training_path.read_text())
  test = json.loads(test_path.read_text())
  classes = training["classes"]
  if (
    len(classes) != 162 or training.get("testImagesIncluded") is not False
    or test.get("classes") != classes
    or test.get("trainingManifestSha256") != sha256(training_path)
  ):
    raise ValueError("Require the frozen original 162-target snapshot.")
  base_rows = training["samples"] + test["samples"]
  class_by_id = {row["speciesId"]: row for row in classes}
  train_counts = Counter(
    row["speciesId"] for row in base_rows if row["split"] == "train"
  )
  missing_train = {row["speciesId"] for row in classes if not train_counts[row["speciesId"]]}
  if len(missing_train) != 17:
    raise ValueError("Expected the reviewed 17-class training gap.")

  duplicates = DuplicateIndex()
  source_images = {}
  for row in base_rows:
    image = base / row["path"]
    if not image.is_file() or sha256(image) != row["imageSha256"]:
      raise ValueError("Base snapshot image changed.")
    duplicates.add(row["imageSha256"], _perceptual(row["differenceHash"]))
    source_images[row["sampleId"]] = image

  candidates, rejected = [], Counter()
  seen_occurrences = set()
  for collection in collections:
    progress = json.loads((collection / "progress.json").read_text())
    rows = json.loads((collection / "candidates.json").read_text())
    if (
      progress.get("stage") != "collection-pass-finished"
      or progress.get("trainingEligible") is not False
      or progress.get("automaticAdmission") is not False
      or progress.get("candidateImages") != len(rows)
    ):
      raise ValueError("GBIF collection is incomplete or already admitted.")
    for candidate in rows:
      sid = candidate["speciesId"]
      target = class_by_id.get(sid)
      occurrence = candidate["gbifOccurrenceKey"]
      if (
        target is None
        or candidate["scientificName"] != target["scientificName"]
        or candidate["chineseName"] != target["chineseName"]
        or candidate.get("trainingEligible") is not False
      ):
        raise ValueError("GBIF target identity changed.")
      if sid not in missing_train:
        rejected["already-had-training-images"] += 1
        continue
      if occurrence in seen_occurrences:
        rejected["duplicate-occurrence"] += 1
        continue
      image = collection / candidate["path"]
      if not image.is_file() or sha256(image) != candidate["imageSha256"]:
        raise ValueError("GBIF candidate image changed.")
      perceptual = _perceptual(candidate["differenceHash"])
      if duplicates.contains(candidate["imageSha256"], perceptual):
        rejected["exact-or-near-duplicate"] += 1
        continue
      seen_occurrences.add(occurrence)
      duplicates.add(candidate["imageSha256"], perceptual)
      author = _author(candidate)
      group = hashlib.sha256(
        f"gbif:{candidate.get('datasetKey')}:{author.casefold()}".encode()
      ).hexdigest()
      sample_id = f"gbif:{occurrence}:{candidate['imageSha256'][:16]}"
      row = {
        "sampleId": sample_id,
        "speciesId": sid,
        "scientificName": target["scientificName"],
        "sourceScientificName": candidate.get(
          "gbifAcceptedSpeciesName", target["scientificName"]
        ),
        "sourceNameEvidence": {
          "matchType": "gbif-exact-species-or-explicit-accepted-synonym",
          "queryScientificName": target["scientificName"],
          "matchedScientificName": candidate.get("gbifMatchedScientificName"),
          "acceptedScientificName": candidate.get("gbifAcceptedSpeciesName"),
          "queryTaxonKey": candidate.get(
            "gbifQueryTaxonKey", candidate.get("gbifTaxonKey")
          ),
          "acceptedTaxonKey": candidate.get(
            "gbifAcceptedTaxonKey", candidate.get("gbifTaxonKey")
          ),
          "confidence": 100,
        },
        "observationId": f"gbif-occurrence:{occurrence}",
        "photoId": candidate["candidateId"],
        "observerGroup": group,
        "split": group_split(group),
        "sourceUrl": candidate.get("sourceReferences"),
        "imageUrl": candidate["sourceImageUrl"],
        "license": _license(candidate["mediaLicense"]),
        "attribution": author,
        "licenseUrl": candidate["mediaLicense"],
        "labelSource": candidate["labelEvidence"],
        "retrievedAt": progress["completedAt"],
        "path": "",
        "imageSha256": candidate["imageSha256"],
        "differenceHash": f"{perceptual:016x}",
        "modifications": "EXIF removed; RGB JPEG; maximum edge 2048px",
        "basisOfRecord": candidate.get("basisOfRecord"),
        "datasetKey": candidate.get("datasetKey"),
        "trainingEligible": True,
        "admissionStatus": "admitted-after-exact-taxonomy-license-dedup-audit",
      }
      candidates.append((row, image))

  # A global source group normally keeps its hash-derived split. If a class would
  # otherwise still have no train image, move one whole group to train everywhere.
  by_species = {}
  for row, _ in candidates:
    by_species.setdefault(row["speciesId"], []).append(row)
  group_overrides = set()
  for sid in sorted(missing_train):
    rows = by_species.get(sid, [])
    if not rows:
      continue
    if not any(row["split"] == "train" for row in rows):
      group_overrides.add(sorted(row["observerGroup"] for row in rows)[0])
  for row, _ in candidates:
    if row["observerGroup"] in group_overrides:
      row["split"] = "train"

  all_rows = base_rows + [row for row, _ in candidates]
  counts = {
    sid: Counter(row["split"] for row in all_rows if row["speciesId"] == sid)
    for sid in class_by_id
  }
  unsupported = [sid for sid, value in counts.items() if not value["train"]]
  if unsupported:
    names = [class_by_id[sid]["scientificName"] for sid in unsupported]
    raise ValueError(f"GBIF sources still lack training images for: {names}")

  output.mkdir(parents=True)
  (output / "images").mkdir()
  copied = []
  for row in base_rows:
    relative = f"images/{row['imageSha256']}.jpg"
    destination = output / relative
    if not destination.exists():
      copy_snapshot_file(source_images[row["sampleId"]], destination, reflink=reflink)
    copied.append({**row, "path": relative})
  for row, source in candidates:
    relative = f"images/{row['imageSha256']}.jpg"
    destination = output / relative
    if not destination.exists():
      copy_snapshot_file(source, destination, reflink=reflink)
    if sha256(destination) != row["imageSha256"]:
      raise ValueError("GBIF snapshot copy changed.")
    copied.append({**row, "path": relative})
  copied.sort(key=lambda row: (row["split"], row["speciesId"], row["sampleId"]))
  train_validation = [row for row in copied if row["split"] != "test"]
  heldout = [row for row in copied if row["split"] == "test"]
  write_json(output / "species.json", {
    "createdAt": now(), "policy": "target162-gbif-exact-supplement-v1",
    "species": classes, "requestedClassCount": 162,
  })
  write_json(output / "train-validation.json", {
    "createdAt": now(), "classes": classes,
    "targetSpeciesIds": [row["speciesId"] for row in classes],
    "samples": train_validation, "seed": 20260930,
    "pretrainedWeightsAllowed": True, "pretrainedClassifierRowsAllowed": True,
    "testImagesIncluded": False, "automaticDeployment": False,
    "scope": "162 targets with exact-taxonomy public GBIF supplementation.",
  })
  write_json(output / "test-heldout.json", {
    "createdAt": now(), "classes": classes, "samples": heldout,
    "trainingManifestSha256": sha256(output / "train-validation.json"),
    "independentFinalBenchmark": False,
    "note": "Development holdout; not the final independent acceptance set.",
  })
  write_json(output / "snapshot.json", {
    "createdAt": now(), "format": 1,
    "policy": "target162-gbif-exact-supplement-v1",
    "requestedClassCount": 162, "trainedClassCount": 162,
    "baseImages": len(base_rows), "addedImages": len(candidates),
    "imageCount": len(copied), "rejected": dict(rejected),
    "counts": {sid: dict(value) for sid, value in counts.items()},
    "groupOverridesToTrain": sorted(group_overrides),
    "allClassesHaveTrainingEvidence": True,
    "validationGoalNotYetEvaluated": True,
    "independentFinalBenchmark": False, "automaticDeployment": False,
    "sources": [{
      "path": str(collection),
      "progressSha256": sha256(collection / "progress.json"),
      "candidatesSha256": sha256(collection / "candidates.json"),
    } for collection in collections],
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--collection", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--copy", action="store_true")
  args = parser.parse_args()
  build(
    args.base.resolve(), [path.resolve() for path in args.collection],
    args.output.resolve(), reflink=not args.copy,
  )


if __name__ == "__main__":
  main()
