"""Prepare an explicitly bounded Chinese-species experiment without reviving the old ranking."""

import argparse
import json
import shutil
from pathlib import Path

from yizhu_server.catalog import PlantCatalog
from yizhu_server.training_classifier import (
  eligible_species,
  encoder_identity,
  feature_key,
  read_feature,
  validate_samples,
)
from yizhu_server.training_data import load_samples, now, sha256, write_json

CHINA_REGIONS = {"东北", "华北", "华东", "华中", "华南", "西南", "西北", "全国"}


def select_classes(rows, pool, catalog):
  """Use exact stable-ID/scientific matches with regional survey evidence, not Chinese names alone."""
  by_id = {}
  for candidate in pool:
    match = candidate.get("catalogMatch") or {}
    if (
      not match.get("speciesId") or candidate.get("scientificLabel") != match.get("scientificName")
      or not candidate.get("sourceRecordIds") or not candidate.get("sourceRegions")
      or not set(candidate["sourceRegions"]) <= CHINA_REGIONS
      or not candidate.get("sourceWorks") or candidate.get("rankStatus") != "种"
      or not set(candidate.get("sourceTypes", [])) & {"survey-wild", "survey-cultivated"}
    ):
      continue
    by_id.setdefault(match["speciesId"], []).append(candidate)
  targets = []
  for identifier in sorted({row["speciesId"] for row in rows}):
    candidates = by_id.get(identifier, [])
    item = catalog.get(identifier)
    if len(candidates) != 1 or item is None:
      continue
    candidate = candidates[0]
    if (
      item["scientificName"] != candidate["scientificLabel"]
      or not item.get("chineseName")
      or any(row["scientificName"] != item["scientificName"]
             for row in rows if row["speciesId"] == identifier)
    ):
      continue
    targets.append({
      "speciesId": identifier, "scientificName": item["scientificName"],
      "chineseName": item["chineseName"], "chineseAliases": item.get("chineseAliases", []),
      "taxonomy": item.get("taxonomy"), "taxonomyChinese": item.get("taxonomyChinese", {}),
      "regionalEvidence": {
        key: candidate[key]
        for key in ("candidateId", "sourceRecordIds", "sourceRegions", "sourceTypes", "sourceWorks")
      },
      "scope": "regional-source-supported-experiment-not-national-commonness-ranking",
    })
  eligible = eligible_species(rows, targets)
  return sorted(eligible, key=lambda row: (row["scientificName"], row["speciesId"]))


def prepare(source, pool_path, catalog_path, output):
  if output.exists():
    raise ValueError("Training dataset already exists; do not overwrite a frozen version.")
  catalog = PlantCatalog(catalog_path)
  if catalog.metadata.get("storageSchemaVersion") != 2:
    raise ValueError("Migrate the catalog before preparing this training dataset.")
  rows = load_samples(source)
  pool = json.loads(pool_path.read_text())
  classes = select_classes(rows, pool, catalog)
  if len(classes) < 6:
    raise ValueError("Need at least six source-supported classes for a nontrivial Top-5 trial.")
  accepted = {row["speciesId"] for row in classes}
  rows = [row for row in rows if row["speciesId"] in accepted]
  validate_samples(rows, classes)
  # Validate source bytes before creating the destination.
  for row in rows:
    path = source / row["path"]
    if not path.resolve().is_relative_to((source / "images").resolve()):
      raise ValueError("Cached photo escapes public dataset.")
    if sha256(path) != row["imageSha256"]:
      raise ValueError("Cached photo bytes changed.")
  output.mkdir(parents=True)
  (output / "images").mkdir()
  (output / "samples").mkdir()
  (output / "features").mkdir()
  identity = encoder_identity()
  reused = 0
  for row in rows:
    shutil.copyfile(source / row["path"], output / row["path"])
    key = feature_key(row, identity)
    cached = source / "features" / f"{key}.npy"
    if cached.exists():
      read_feature(cached, identity["dimension"])
      shutil.copyfile(cached, output / "features" / cached.name)
      reused += 1
  write_json(output / "samples/public-cache.json", {"samples": rows})
  document = {
    "createdAt": now(), "policy": "regional-exact-id-public-cache-species-v1",
    "purpose": "Chinese-species classification, separate from plant-presence classification",
    "catalogSha256": sha256(catalog_path), "catalogStorageSchemaVersion": 2,
    "candidatePoolSha256": sha256(pool_path), "species": classes,
    "requestedClassCount": len(classes), "images": len(rows), "reusedFeatures": reused,
    "authorization": "User requested continued Chinese-species training after the locale refactor.",
    "limitations": [
      "Small regional-evidence experiment; not all Chinese plants or a national commonness ranking.",
      "The source candidate pool remains pending broad taxonomy/regional review; not globally approved.",
      "Public photo labels are community research-grade, not expert-audited ground truth.",
      "Existing observer splits preserved; existing public test examples have been evaluated before.",
      "Worldwide photos of the chosen species, not a geographically China-only image collection.",
      "Old paused dataset is read-only; its training/collection has not been resumed.",
    ],
  }
  write_json(output / "species.json", document)
  return document


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True, type=Path)
  parser.add_argument("--pool", required=True, type=Path)
  parser.add_argument("--catalog", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = prepare(args.source, args.pool, args.catalog, args.output)
  print(json.dumps({
    "classes": result["requestedClassCount"], "images": result["images"],
    "reusedFeatures": result["reusedFeatures"],
    "species": [{"id": row["speciesId"], "scientificName": row["scientificName"],
                 "chineseName": row["chineseName"]} for row in result["species"]],
    "scope": result["limitations"][0],
  }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
