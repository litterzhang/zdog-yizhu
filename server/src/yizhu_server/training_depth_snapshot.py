"""Audit parallel train-only supplements and freeze one reproducible 82-class snapshot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from yizhu_server.snapshot_files import copy_snapshot_file
from yizhu_server.training_classifier import validate_samples
from yizhu_server.training_data import (
  DuplicateIndex,
  difference_hash,
  group_split,
  now,
  sha256,
  write_json,
)

POLICY = "audited-depth82-train-supplement-v1"
SPLITS = {"train": 70, "validation": 15, "test": 15}


def _read_collection(root, *, require_images=True):
  plan = json.loads((root / "plan.json").read_text())
  progress = json.loads((root / "progress.json").read_text())
  if (
    plan.get("policy") != "train-only-82-cached-leads-fresh-research-cc0-ccby-v1"
    or progress.get("stage") != "collection-pass-complete"
    or progress.get("trainingImagesAdmitted") != 0
    or progress.get("frozenSplitsChanged") is not False
    or progress.get("trainingStarted") is not False
  ):
    raise ValueError("Collection is incomplete or already altered training state.")
  rows = []
  for path in sorted((root / "samples").glob("*.json")):
    document = json.loads(path.read_text())
    if len(document.get("samples", [])) != 1:
      raise ValueError("Each supplemental file must contain exactly one sample.")
    row = document["samples"][0]
    image = root / row["path"]
    if (
      row.get("split") != "train" or group_split(row["observerGroup"]) != "train"
      or row.get("license") not in {"cc0", "cc-by"}
      or not row.get("attribution") or row.get("trainingEligible") is not False
      or row.get("admissionStatus") != "pending-consolidated-dataset-audit"
      or row.get("automatedSourceChecksPassed") is not True
      or row.get("sourceQualityGrade") != "research"
      or (require_images and (not image.is_file() or sha256(image) != row["imageSha256"]))
    ):
      raise ValueError("Supplemental sample provenance or image changed.")
    rows.append((row, image))
  if len(rows) != progress["downloadedImages"]:
    raise ValueError("Downloaded image count and metadata files differ.")
  candidates = {row["sampleId"]: row for row in plan["candidates"]}
  if len(candidates) != len(plan["candidates"]):
    raise ValueError("Duplicate candidates in collection plan.")
  for row, _ in rows:
    lead = candidates.get(row["sampleId"])
    if lead is None or any(row[k] != lead[k] for k in (
      "speciesId", "scientificName", "photoId", "observationId", "observerGroup",
    )):
      raise ValueError("Downloaded sample differs from its fixed lead.")
  return plan, progress, rows


def audit_metadata(collections, output):
  """Cross-host identity/dHash/author audit before any feature extraction or training."""
  if output.exists():
    raise ValueError("Use a new audit output.")
  loaded = [_read_collection(root, require_images=False) for root in collections]
  plans = [item[0] for item in loaded]
  if any(plan["classes"] != plans[0]["classes"] for plan in plans[1:]):
    raise ValueError("Collection class order differs.")
  source_plan = plans[0].get("partition", {}).get("sourcePlanSha256")
  if (
    len(collections) != 2 or source_plan is None
    or {p.get("partition", {}).get("index") for p in plans} != {1, 2}
    or any(p.get("partition", {}).get("count") != 2 for p in plans)
    or any(p.get("partition", {}).get("sourcePlanSha256") != source_plan for p in plans)
    or plans[0]["baseline"] != plans[1]["baseline"]
  ):
    raise ValueError("Require both partitions of one source plan.")
  groups = [
    {(r["speciesId"], r["observerGroup"]) for r in p["candidates"]} for p in plans
  ]
  if groups[0] & groups[1]:
    raise ValueError("Species-author group spans collection hosts.")
  duplicates = DuplicateIndex()
  author_counts = Counter()
  seen_samples, seen_observations, seen_photos = set(), set(), set()
  for row in plans[0]["baseline"]:
    duplicates.add(row["imageSha256"], int(row["differenceHash"], 16))
    author_counts[(row["speciesId"], row["observerGroup"])] += 1
    seen_samples.add(row["sampleId"])
    seen_observations.add(row["observationId"])
    seen_photos.add(row["photoId"])
  accepted, rejected = [], Counter()
  rows = sorted(
    [(root.name, row) for root, (_, _, entries) in zip(collections, loaded, strict=True)
     for row, _ in entries],
    key=lambda pair: pair[1]["sampleId"],
  )
  for source, row in rows:
    key = (row["speciesId"], row["observerGroup"])
    if (
      row["sampleId"] in seen_samples or row["observationId"] in seen_observations
      or row["photoId"] in seen_photos
    ):
      rejected["identity-duplicate"] += 1
      continue
    if author_counts[key] >= 3:
      rejected["global-author-cap"] += 1
      continue
    post = int(row["differenceHash"], 16)
    pre = int(row["preEncodingDifferenceHash"], 16)
    if (
      duplicates.contains(row["imageSha256"], post)
      or duplicates.contains(row["imageSha256"], pre)
    ):
      rejected["cross-pool-exact-or-near-duplicate"] += 1
      continue
    accepted.append({
      "sourceCollection": source, "sampleId": row["sampleId"],
      "speciesId": row["speciesId"], "photoId": row["photoId"],
      "observationId": row["observationId"], "observerGroup": row["observerGroup"],
      "imageSha256": row["imageSha256"], "differenceHash": row["differenceHash"],
      "preEncodingDifferenceHash": row["preEncodingDifferenceHash"],
      "path": row["path"], "eligibleForFeatureExtraction": True,
      "trainingAdmitted": False,
    })
    duplicates.add(row["imageSha256"], post)
    duplicates.add(row["imageSha256"], pre)
    author_counts[key] += 1
    seen_samples.add(row["sampleId"])
    seen_observations.add(row["observationId"])
    seen_photos.add(row["photoId"])
  report = {
    "checkedAt": now(), "policy": "cross-host-depth82-metadata-audit-v1",
    "sourcePlanSha256": source_plan, "classes": plans[0]["classes"],
    "downloadedCandidates": len(rows), "acceptedForFeatureExtraction": len(accepted),
    "coveredSpecies": len({r["speciesId"] for r in accepted}),
    "rejected": dict(rejected), "accepted": accepted,
    "imagesDecodedDuringThisAudit": False,
    "collectorAlreadyDecodedAndRehashedFinalPixels": True,
    "trainingAdmitted": False, "trainingStarted": False,
    "validationOrTestChanged": False,
    "sourceFiles": {
      root.name: {
        "planSha256": sha256(root / "plan.json"),
        "progressSha256": sha256(root / "progress.json"),
      } for root in collections
    },
  }
  write_json(output, report)
  return report


def _clean_pixels(path):
  """Recompute final-pixel dHash; do not trust only pre-download metadata."""
  from yizhu_server.recognition import read_photo

  with read_photo(path) as image:
    return difference_hash(image)


def build(base, collections, output, *, reflink=True):
  if output.exists():
    raise ValueError("Use a new snapshot directory; never overwrite.")
  training_path = base / "train-validation.json"
  test_path = base / "test-heldout.json"
  training = json.loads(training_path.read_text())
  test = json.loads(test_path.read_text())
  classes = training["classes"]
  if (
    len(classes) != 82 or training.get("testImagesIncluded") is not False
    or test["classes"] != classes or test["trainingManifestSha256"] != sha256(training_path)
    or any(r["split"] not in {"train", "validation"} for r in training["samples"])
    or any(r["split"] != "test" for r in test["samples"])
  ):
    raise ValueError("Require the frozen 82-class development data.")
  original = training["samples"] + test["samples"]
  validate_samples(original, classes)
  source_images = {}
  for row in original:
    path = base / row["path"]
    if not path.is_file() or sha256(path) != row["imageSha256"]:
      raise ValueError("Original image or checksum changed.")
    source_images[row["sampleId"]] = path

  loaded = [_read_collection(root) for root in collections]
  plans = [item[0] for item in loaded]
  if any(plan["classes"] != classes for plan in plans):
    raise ValueError("Collection class order differs from the frozen model labels.")
  source_plan = plans[0].get("partition", {}).get("sourcePlanSha256")
  if (
    len(collections) != 2 or source_plan is None
    or {p.get("partition", {}).get("index") for p in plans} != {1, 2}
    or any(p.get("partition", {}).get("count") != 2 for p in plans)
    or any(p.get("partition", {}).get("sourcePlanSha256") != source_plan for p in plans)
    or plans[0]["baseline"] != plans[1]["baseline"]
  ):
    raise ValueError("Require both non-overlapping partitions from one source plan.")
  groups = [
    {(r["speciesId"], r["observerGroup"]) for r in p["candidates"]} for p in plans
  ]
  candidate_ids = [{r["sampleId"] for r in p["candidates"]} for p in plans]
  if groups[0] & groups[1] or candidate_ids[0] & candidate_ids[1]:
    raise ValueError("Parallel partitions overlap by sample or species-author group.")

  baseline = plans[0]["baseline"]
  baseline_identity = {
    (r["sampleId"], r["speciesId"], r["split"], r["imageSha256"]) for r in baseline
  }
  if any(
    (r["sampleId"], r["speciesId"], r["split"], r["imageSha256"]) not in baseline_identity
    for r in original
  ):
    raise ValueError("Dedup baseline does not cover all frozen development images.")
  duplicates = DuplicateIndex()
  author_counts = Counter()
  seen_observations, seen_photos, seen_samples = set(), set(), set()
  for row in baseline:
    duplicates.add(row["imageSha256"], int(row["differenceHash"], 16))
    author_counts[(row["speciesId"], row["observerGroup"])] += 1
    seen_observations.add(row["observationId"])
    seen_photos.add(row["photoId"])
    seen_samples.add(row["sampleId"])

  accepted, rejected = [], Counter()
  # Deterministic order independent of transfer/host timing.
  candidates = sorted(
    [pair for _, _, rows in loaded for pair in rows],
    key=lambda pair: pair[0]["sampleId"],
  )
  for row, image in candidates:
    key = (row["speciesId"], row["observerGroup"])
    if (
      row["sampleId"] in seen_samples or row["observationId"] in seen_observations
      or row["photoId"] in seen_photos
    ):
      rejected["identity-duplicate"] += 1
      continue
    if author_counts[key] >= 3:
      rejected["global-author-cap"] += 1
      continue
    pixel_hash = _clean_pixels(image)
    recorded_hash = int(row["differenceHash"], 16)
    pre_hash = int(row["preEncodingDifferenceHash"], 16)
    if pixel_hash != recorded_hash:
      raise ValueError("Final-pixel dHash differs from downloaded metadata.")
    if (
      duplicates.contains(row["imageSha256"], pixel_hash)
      or duplicates.contains(row["imageSha256"], pre_hash)
    ):
      rejected["cross-pool-exact-or-near-duplicate"] += 1
      continue
    admitted = {
      **row, "trainingEligible": True,
      "admissionStatus": "admitted-after-consolidated-dataset-audit",
      "consolidatedAuditAt": now(),
    }
    accepted.append((admitted, image))
    duplicates.add(row["imageSha256"], pixel_hash)
    duplicates.add(row["imageSha256"], pre_hash)
    author_counts[key] += 1
    seen_samples.add(row["sampleId"])
    seen_observations.add(row["observationId"])
    seen_photos.add(row["photoId"])

  all_rows = original + [r for r, _ in accepted]
  validate_samples(all_rows, classes)
  counts = {sid: Counter() for sid in (r["speciesId"] for r in classes)}
  for row in all_rows:
    counts[row["speciesId"]][row["split"]] += 1
  if any(any(counts[sid][split] < minimum for split, minimum in SPLITS.items())
         for sid in counts):
    raise ValueError("Every class must retain the original train/validation/test coverage.")
  if any(
    counts[sid]["validation"] != 15 or counts[sid]["test"] != 15 for sid in counts
  ):
    raise ValueError("Supplementation must not change frozen validation/test counts.")

  output.mkdir(parents=True)
  (output / "images").mkdir()
  copied_rows = []
  for row, source in [(r, source_images[r["sampleId"]]) for r in original] + accepted:
    relative = f"images/{row['imageSha256']}.jpg"
    destination = output / relative
    if not destination.exists():
      copy_snapshot_file(source, destination, reflink=reflink)
    if sha256(destination) != row["imageSha256"]:
      raise ValueError("Snapshot image checksum changed.")
    copied_rows.append({**row, "path": relative})
  copied_rows.sort(key=lambda r: (r["split"], r["speciesId"], r["sampleId"]))
  species = {
    "createdAt": now(), "policy": POLICY, "species": classes,
    "splitTargets": SPLITS, "requestedClassCount": len(classes),
    "sourcePlanSha256": source_plan,
  }
  write_json(output / "species.json", species)
  write_json(output / "samples/frozen.json", {"samples": copied_rows})
  train_validation = [r for r in copied_rows if r["split"] != "test"]
  heldout = [r for r in copied_rows if r["split"] == "test"]
  write_json(output / "train-validation.json", {
    "createdAt": now(), "classes": classes,
    "targetSpeciesIds": [r["speciesId"] for r in classes],
    "samples": train_validation, "seed": 20260930,
    "pretrainedWeightsAllowed": True, "pretrainedClassifierRowsAllowed": True,
    "testImagesIncluded": False, "automaticDeployment": False,
    "scope": "82-class development snapshot with audited train-only supplementation.",
  })
  write_json(output / "test-heldout.json", {
    "createdAt": now(), "classes": classes, "samples": heldout,
    "trainingManifestSha256": sha256(output / "train-validation.json"),
    "independentFinalBenchmark": False,
    "note": "Unchanged historical development test; already viewed, never final acceptance.",
  })
  write_json(output / "snapshot.json", {
    "createdAt": now(), "format": 1, "policy": POLICY,
    "speciesSha256": sha256(output / "species.json"),
    "samplesSha256": sha256(output / "samples/frozen.json"),
    "requestedClassCount": len(classes), "readyClassCount": len(classes),
    "imageCount": len(copied_rows), "addedTrainingImages": len(accepted),
    "originalImages": len(original), "rejectedSupplemental": dict(rejected),
    "counts": {sid: dict(value) for sid, value in counts.items()},
    "validationAndTestUnchanged": True, "testUsedForSelection": False,
    "independentFinalBenchmark": False, "automaticTraining": False,
    "automaticDeployment": False,
    "collectionProgressSha256": {
      root.name: sha256(root / "progress.json") for root in collections
    },
  })
  return json.loads((output / "snapshot.json").read_text())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--audit-metadata", action="store_true")
  parser.add_argument("--base", type=Path)
  parser.add_argument("--collection", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--copy", action="store_true",
                      help="Use full copies; default requires filesystem reflinks.")
  args = parser.parse_args()
  collections = [p.resolve() for p in args.collection]
  if args.audit_metadata:
    result = audit_metadata(collections, args.output.resolve())
  elif args.base:
    result = build(
      args.base.resolve(), collections, args.output.resolve(), reflink=not args.copy,
    )
  else:
    parser.error("--base is required unless --audit-metadata is used")
  print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
  main()
