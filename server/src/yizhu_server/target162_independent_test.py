"""Audit new public candidates into a model-frozen independent target test manifest."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def _perceptual(value):
  return int(value, 16) if isinstance(value, str) else int(value)


def _author(candidate):
  return str(
    candidate.get("mediaCreator") or candidate.get("mediaRightsHolder")
    or candidate.get("recordedBy") or candidate.get("attribution")
    or candidate.get("datasetKey") or candidate.get("source")
  ).strip()


def _source_group(candidate):
  if candidate.get("authorGroup"):
    return candidate["authorGroup"]
  return hashlib.sha256(
    f"gbif:{candidate.get('datasetKey')}:{_author(candidate).casefold()}".encode()
  ).hexdigest()


def audit(snapshot, model_metadata, collections, output, *, per_species=5,
          require_images=True, require_unseen_source_groups=True,
          exclude_manifests=(), allow_known_occurrences=False):
  if output.exists() or not 1 <= per_species <= 20:
    raise ValueError("Use a new output and bounded independent sample target.")
  training = json.loads((snapshot / "train-validation.json").read_text())
  development_test = json.loads((snapshot / "test-heldout.json").read_text())
  classes = training["classes"]
  class_count = len(classes)
  if class_count < 2:
    raise ValueError("Independent audit requires a reviewed multi-species target list.")
  model = json.loads(model_metadata.read_text())
  if (
    model.get("status") != "trained-offline-development-candidate"
    or model.get("plantSet", {}).get("count") != class_count
    or model.get("productionApproved") is not False
  ):
    raise ValueError("Freeze one complete offline model before independent test collection.")
  class_by_id = {row["speciesId"]: row for row in classes}
  known = training["samples"] + development_test["samples"]
  exact_hashes = set()
  species_perceptual_hashes = defaultdict(set)
  known_occurrences, known_urls, known_groups = set(), set(), set()
  for row in known:
    exact_hashes.add(row["imageSha256"])
    species_perceptual_hashes[row["speciesId"]].add(
      _perceptual(row["differenceHash"])
    )
    known_groups.add(row["observerGroup"])
    observation = str(row.get("observationId", ""))
    if observation.startswith("gbif-occurrence:"):
      known_occurrences.add(observation.split(":", 1)[1])
    elif observation:
      known_occurrences.add(observation)
    if row.get("imageUrl"):
      known_urls.add(row["imageUrl"])
  excluded_independent_samples = 0
  for path in exclude_manifests:
    previous = json.loads(path.read_text())
    if (
      previous.get("purpose")
      != "new-independent-test-never-used-for-training-or-selection"
      or previous.get("representedSpecies") != len({
        row["speciesId"] for row in previous.get("samples", [])
      })
      or previous.get("modelOutputsReadDuringSelection") is not False
    ):
      raise ValueError("Prior independent exclusion manifest is invalid.")
    for row in previous["samples"]:
      exact_hashes.add(row["imageSha256"])
      species_perceptual_hashes[row["speciesId"]].add(
        _perceptual(row["differenceHash"])
      )
      known_occurrences.add(str(row.get("sourceRecordId")))
      if row.get("sourceImageUrl"):
        known_urls.add(row["sourceImageUrl"])
      excluded_independent_samples += 1

  rejected = Counter()
  known_source_group_selected = 0
  known_occurrence_selected = 0
  accepted_by_species = defaultdict(list)
  seen_occurrences, seen_urls = set(), set()
  independent_group_counts = Counter()
  candidates = []
  for collection in collections:
    progress = json.loads((collection / "progress.json").read_text())
    rows = json.loads((collection / "candidates.json").read_text())
    candidate_count = progress.get(
      "candidateImages", progress.get("downloadedCandidateImages"),
    )
    if (
      progress.get("stage") not in {
        "collection-pass-finished", "collection-partial-finalized",
      }
      or (
        progress.get("stage") == "collection-partial-finalized"
        and progress.get("partialScope") is not True
      )
      or progress.get("trainingEligible") is not False
      or (
        progress.get("automaticAdmission") is not False
        and progress.get("onlineChanged") is not False
      )
      or candidate_count != len(rows)
    ):
      raise ValueError("Independent candidate collection is incomplete.")
    for row in rows:
      target = class_by_id.get(row["speciesId"])
      image = collection / row["path"]
      if target is None:
        rejected["outside-frozen-target"] += 1
        continue
      if (
        row["scientificName"] != target["scientificName"]
        or row.get("trainingEligible") is not False
      ):
        raise ValueError("Independent candidate identity changed.")
      candidates.append((row, image, collection))

  # Deterministic priority: field observations, then non-specimen records, then
  # preserved specimens; stable IDs break ties without consulting model output.
  basis_priority = {
    "HUMAN_OBSERVATION": 0, "MACHINE_OBSERVATION": 1,
    "LIVING_SPECIMEN": 2, "MATERIAL_SAMPLE": 3,
    "PRESERVED_SPECIMEN": 4,
  }
  candidates.sort(key=lambda item: (
    basis_priority.get(item[0].get("basisOfRecord"), 3),
    item[0]["speciesId"], item[0].get("candidateId") or item[0].get("id"),
  ))
  for row, image, collection in candidates:
    sid = row["speciesId"]
    candidate_id = row.get("candidateId") or row.get("id")
    occurrence = row.get("gbifOccurrenceKey", row.get("sourceObservationId", candidate_id))
    url = row.get("sourceImageUrl") or row.get("imageUrl")
    group = _source_group(row)
    perceptual = _perceptual(row["differenceHash"])
    if len(accepted_by_species[sid]) >= per_species:
      rejected["per-species-cap"] += 1
      continue
    occurrence_key = str(occurrence)
    if occurrence_key in seen_occurrences:
      rejected["duplicate-occurrence"] += 1
      continue
    if occurrence_key in known_occurrences:
      if not allow_known_occurrences:
        rejected["known-occurrence"] += 1
        continue
      known_occurrence_selected += 1
    if url in known_urls or url in seen_urls:
      rejected["known-or-duplicate-source-url"] += 1
      continue
    if group in known_groups:
      if require_unseen_source_groups:
        rejected["source-group-used-by-training-or-development"] += 1
        continue
      known_source_group_selected += 1
    if independent_group_counts[(sid, group)] >= 2:
      rejected["independent-per-species-source-group-cap"] += 1
      continue
    if (
      row["imageSha256"] in exact_hashes
      or perceptual in species_perceptual_hashes[sid]
    ):
      rejected["known-exact-or-near-duplicate"] += 1
      continue
    selected = {
      "sampleId": f"independent:{candidate_id}",
      "speciesId": sid, "scientificName": row["scientificName"],
      "chineseName": row["chineseName"],
      "sourceRecordId": occurrence,
      "observerGroup": group, "basisOfRecord": row.get("basisOfRecord"),
      "imageSha256": row["imageSha256"],
      "differenceHash": f"{perceptual:016x}",
      "sourceImageUrl": url,
      "sourceReferences": row.get("sourceReferences") or row.get("sourceUrl"),
      "mediaLicense": row.get("mediaLicense") or row.get("licenseUrl"),
      "attribution": _author(row),
      "sourceCollection": str(collection), "path": str(image.resolve()),
    }
    accepted_by_species[sid].append(selected)
    seen_occurrences.add(occurrence_key)
    seen_urls.add(url)
    independent_group_counts[(sid, group)] += 1
    exact_hashes.add(row["imageSha256"])
    species_perceptual_hashes[sid].add(perceptual)

  selected = [
    row for target in classes for row in accepted_by_species[target["speciesId"]]
  ]
  if require_images:
    for row in selected:
      image = Path(row["path"])
      if not image.is_file() or sha256(image) != row["imageSha256"]:
        raise ValueError(
          f"Selected independent image missing or changed: {row['sampleId']}"
        )
  missing = [
    {
      "speciesId": target["speciesId"], "scientificName": target["scientificName"],
      "chineseName": target["chineseName"],
      "selected": len(accepted_by_species[target["speciesId"]]),
      "missing": per_species - len(accepted_by_species[target["speciesId"]]),
    }
    for target in classes
    if len(accepted_by_species[target["speciesId"]]) < per_species
  ]
  represented = sum(bool(accepted_by_species[target["speciesId"]]) for target in classes)
  field_samples = sum(
    row.get("basisOfRecord") == "HUMAN_OBSERVATION" for row in selected
  )
  evidence_path = next(
    (snapshot / name for name in ("admission.json", "audit.json", "snapshot.json")
     if (snapshot / name).is_file()),
    None,
  )
  if evidence_path is None:
    raise ValueError("Training snapshot evidence is missing.")
  output.mkdir(parents=True)
  write_json(output / "manifest.json", {
    "createdAt": now(), "status": "complete" if not missing else "incomplete",
    "purpose": "new-independent-test-never-used-for-training-or-selection",
    "modelMetadataSha256": sha256(model_metadata),
    "frozenClassifierSha256": model["classifier"]["sha256"],
    "trainingSnapshotEvidenceFile": evidence_path.name,
    "trainingSnapshotSha256": sha256(evidence_path),
    "requestedSpecies": class_count, "targetSamplesPerSpecies": per_species,
    "representedSpecies": represented, "samples": selected,
    "modelOutputsReadDuringSelection": False,
    "excludedPriorIndependentSamples": excluded_independent_samples,
    "excludedIndependentManifestSha256": [
      sha256(path) for path in exclude_manifests
    ],
    "sourceGroupsDisjointFromTraining": require_unseen_source_groups,
    "selectedImagesFromKnownSourceGroups": known_source_group_selected,
    "selectedImagesFromKnownOccurrences": known_occurrence_selected,
    "imagesVerified": require_images,
    "productionChanged": False,
  })
  write_json(output / "audit.json", {
    "createdAt": now(), "candidateImages": len(candidates),
    "selectedImages": len(selected), "representedSpecies": represented,
    "completeSpecies": class_count - len(missing),
    "fieldObservationImages": field_samples,
    "basisOfRecord": dict(Counter(row.get("basisOfRecord") for row in selected)),
    "rejected": dict(rejected), "shortfalls": missing,
    "readyForOneShotEvaluation": not missing,
    "sourceGroupsDisjointFromTraining": require_unseen_source_groups,
    "selectedImagesFromKnownSourceGroups": known_source_group_selected,
    "selectedImagesFromKnownOccurrences": known_occurrence_selected,
    "imagesVerified": require_images,
    "modelOutputsReadDuringSelection": False,
    "excludedPriorIndependentSamples": excluded_independent_samples,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path, required=True)
  parser.add_argument("--model-metadata", type=Path, required=True)
  parser.add_argument("--collection", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--per-species", type=int, default=5)
  parser.add_argument("--metadata-only", action="store_true")
  parser.add_argument("--allow-known-source-groups", action="store_true")
  parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
  parser.add_argument("--allow-known-occurrences", action="store_true")
  args = parser.parse_args()
  audit(
    args.snapshot.resolve(), args.model_metadata.resolve(),
    [path.resolve() for path in args.collection], args.output.resolve(),
    per_species=args.per_species, require_images=not args.metadata_only,
    require_unseen_source_groups=not args.allow_known_source_groups,
    exclude_manifests=[path.resolve() for path in args.exclude_manifest],
    allow_known_occurrences=args.allow_known_occurrences,
  )


if __name__ == "__main__":
  main()
