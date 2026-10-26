"""Bounded train-only supplementation from cached leads, with fresh source checks."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests

from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import (
  DuplicateIndex,
  PublicClient,
  dataset_lock,
  difference_hash,
  group_split,
  load_samples,
  now,
  observer_group,
  require_dataset_active,
  sha256,
  write_json,
)

POLICY = "train-only-82-cached-leads-fresh-research-cc0-ccby-v1"
PHOTO_HOST = "inaturalist-open-data.s3.amazonaws.com"
LICENSES = {"cc0", "cc-by"}


def prepare(repo, destination):
  """Build a small portable plan; no image decoding or test predictions are needed."""
  if destination.exists():
    raise ValueError("Use a new plan file.")
  manifest_path = repo / "server/.cache/training/china-landscape-scratch-batch-002/train-validation.json"
  manifest = json.loads(manifest_path.read_text())
  classes = {r["speciesId"]: r for r in manifest["classes"]}
  if len(classes) != 82 or manifest.get("testImagesIncluded") is not False:
    raise ValueError("Require the frozen 82-class train/validation manifest.")
  inventory_dir = repo / "server/.artifacts/landscape-union-249-20260925"
  inventory = json.loads((inventory_dir / "verified-image-files.json").read_text())["files"]
  pools = json.loads((inventory_dir / "verified-image-inventory.json").read_text())[
    "scannedPublicDatasets"
  ]
  by_path = {}
  for relative in pools:
    root = repo / relative
    require_dataset_active(root)
    for row in load_samples(root):
      by_path[str(root / row["path"])] = row
  baseline = []
  for item in inventory:
    row = by_path[str(repo / item["path"])]
    if any(row[k] != item[k] for k in ("speciesId", "split", "license")):
      raise ValueError("Exclusion inventory identity changed.")
    if row["imageSha256"] != item["sha256"]:
      raise ValueError("Exclusion inventory checksum changed.")
    baseline.append({k: row[k] for k in (
      "sampleId", "speciesId", "photoId", "observationId", "observerGroup",
      "split", "imageSha256", "differenceHash",
    )})
  frozen_rows = {
    (r["sampleId"], r["speciesId"], r["split"], r["imageSha256"]) for r in baseline
  }
  if any(
    (r["sampleId"], r["speciesId"], r["split"], r["imageSha256"]) not in frozen_rows
    for r in manifest["samples"]
  ):
    raise ValueError("Exclusion metadata does not cover the frozen train/validation data.")
  leads_path = repo / "server/.artifacts/landscape-training-depth-20260925/candidates.json"
  leads = json.loads(leads_path.read_text())
  cache, selected, mappings = {}, [], {}
  for lead in leads:
    if lead["speciesId"] not in classes:
      continue
    path = (repo / lead["metadataFile"]).resolve()
    training_root = (repo / "server/.cache/training").resolve()
    if not path.is_relative_to(training_root) or path.parent.name != "photo-metadata":
      raise ValueError("Cache path outside public training metadata.")
    require_dataset_active(path.parent.parent)
    if path not in cache:
      cache[path] = json.loads(path.read_text())["samples"]
    matches = [r for r in cache[path] if r["sampleId"] == lead["sampleId"]
               and r["speciesId"] == lead["speciesId"]]
    if len(matches) != 1:
      raise ValueError("Ambiguous cached lead.")
    row = matches[0]
    if (
      row["split"] != "train" or group_split(row["observerGroup"]) != "train"
      or row["scientificName"] != classes[row["speciesId"]]["scientificName"]
      or any(row[k] != lead[k] for k in (
        "photoId", "observationId", "observerGroup", "scientificName",
      ))
    ):
      raise ValueError("Lead identity or split differs.")
    sid = row["speciesId"]
    evidence = row.get("sourceNameEvidence", {})
    taxon_id = evidence.get("sourceTaxonId")
    if taxon_id is None:
      mapping_path = repo / "server/.cache/training/china-landscape-public-249/taxon-matches"
      mapping = json.loads((mapping_path / (hashlib.sha256(sid.encode()).hexdigest() + ".json"))
                           .read_text())
      if (
        mapping["status"] != "matched" or mapping["speciesId"] != sid
        or mapping["scientificName"] != row["scientificName"]
      ):
        raise ValueError("Missing explicit taxon match.")
      taxon_id = mapping["inatTaxonId"]
    species = {
      **classes[sid], "inatTaxonId": taxon_id,
      "sourceScientificName": row.get("sourceScientificName", row["scientificName"]),
      "sourceNameEvidence": evidence,
    }
    if sid in mappings and mappings[sid] != species:
      # Evidence detail may vary, but source identity must not.
      for key in ("inatTaxonId", "sourceScientificName"):
        if mappings[sid][key] != species[key]:
          raise ValueError("Conflicting source taxonomy.")
    mappings[sid] = species
    selected.append({**lead, "species": species})
  # Round-robin all species, independent of validation scores or confusion counts.
  grouped = defaultdict(list)
  for row in selected:
    grouped[row["speciesId"]].append(row)
  ordered = []
  for i in range(max(map(len, grouped.values()))):
    for sid in sorted(grouped):
      if i < len(grouped[sid]):
        ordered.append(grouped[sid][i])
  plan = {
    "createdAt": now(), "policy": POLICY, "classes": list(classes.values()),
    "manifestSha256": sha256(manifest_path), "leadsSha256": sha256(leads_path),
    "inventorySha256": sha256(inventory_dir / "verified-image-files.json"),
    "candidates": ordered, "baseline": baseline,
    "scope": "Only existing cached train leads for 82 classes; no new taxonomy or split.",
    "perSpeciesAuthorCap": 3, "maximumNewImages": len(ordered),
    "testImagesDecoded": False, "testPredictionsRead": False,
    "automaticTraining": False, "automaticDeployment": False,
    "deduplicationLimit": (
      "Cross-pool SHA and recorded dHash checks; legacy hashes may precede JPEG encoding. "
      "New images get both pre/post-encoding hashes. Consolidated audit required before training."
    ),
  }
  write_json(destination, plan)
  return plan


def refresh_candidate(payload, lead):
  """Revalidate the exact observation/photo; never silently switch to another photo."""
  observations = payload.get("results", [])
  if len(observations) != 1 or observations[0].get("id") != lead["observationId"]:
    return None
  observation = observations[0]
  species = lead["species"]
  taxon = observation.get("taxon") or {}
  user_id = (observation.get("user") or {}).get("id")
  if (
    observation.get("quality_grade") != "research"
    or taxon.get("id") != species["inatTaxonId"]
    or taxon.get("name") != species.get("sourceScientificName", species["scientificName"])
    or taxon.get("rank") != "species"
    or type(user_id) is not int or user_id < 1
  ):
    return None
  photos = [p for p in observation.get("photos", []) if p.get("id") == lead["photoId"]]
  if len(photos) != 1:
    return None
  photo = photos[0]
  url = urlparse(photo.get("url", ""))
  if (
    photo.get("license_code") not in LICENSES or photo.get("hidden")
    or photo.get("flags") or not photo.get("attribution")
    or url.scheme != "https" or url.netloc != PHOTO_HOST or url.query or url.fragment
    or Path(url.path).suffix.lower() not in {".jpg", ".jpeg", ".png"}
  ):
    return None
  group = observer_group(user_id)
  row = {
    "sampleId": f"inat:{observation['id']}:{photo['id']}",
    "speciesId": species["speciesId"], "scientificName": species["scientificName"],
    **({
      "sourceScientificName": species["sourceScientificName"],
      "sourceNameEvidence": species.get("sourceNameEvidence", {}),
    } if species.get("sourceScientificName") else {}),
    "observationId": observation["id"], "photoId": photo["id"],
    "observerGroup": group, "split": group_split(group),
    "sourceUrl": f"https://www.inaturalist.org/observations/{observation['id']}",
    "imageUrl": f"https://{PHOTO_HOST}/photos/{photo['id']}/medium{Path(url.path).suffix}",
    "license": photo["license_code"], "attribution": photo["attribution"],
    "licenseUrl": (
      "https://creativecommons.org/publicdomain/zero/1.0/"
      if photo["license_code"] == "cc0"
      else "https://creativecommons.org/licenses/by/4.0/"
    ),
    "labelSource": "iNaturalist research grade (not expert-audited gold label)",
    "retrievedAt": now(),
  }
  if row["split"] != "train" or any(row[k] != lead[k] for k in (
    "sampleId", "speciesId", "photoId", "observationId", "observerGroup",
  )):
    return None
  return row


def collect(plan_path, output, *, client=None):
  """One bounded pass. Errors stop durably, without automatic restarts or bypasses."""
  from yizhu_server.recognition import read_photo

  client = PublicClient() if client is None else client
  plan = json.loads(plan_path.read_text())
  if plan.get("policy") != POLICY or not plan.get("candidates"):
    raise ValueError("Require a prepared bounded collection plan.")
  if output.exists():
    raise ValueError("Use a new output directory; no implicit resume.")
  output.mkdir(parents=True)
  reserve_space(output, 32 * 1024**2)
  write_json(output / "plan.json", plan)
  index, seen_photos, seen_observations = DuplicateIndex(), set(), set()
  author_counts = Counter()
  for row in plan["baseline"]:
    index.add(row["imageSha256"], int(row["differenceHash"], 16))
    seen_photos.add(row["photoId"])
    seen_observations.add(row["observationId"])
    author_counts[(row["speciesId"], row["observerGroup"])] += 1
  allowed = {r["speciesId"] for r in plan["classes"]}
  frozen_sha = {
    r["imageSha256"] for r in plan["baseline"] if r["split"] in {"validation", "test"}
  }
  rows, rejected, completed = [], Counter(), 0
  (output / "images").mkdir()

  def progress(stage):
    status = {
      "updatedAt": now(), "stage": stage, "checkedCandidates": completed,
      "totalCandidates": len(plan["candidates"]), "downloadedImages": len(rows),
      "coveredSpecies": len({r["speciesId"] for r in rows}), "rejections": dict(rejected),
      "trainingImagesAdmitted": 0, "frozenSplitsChanged": False, "trainingStarted": False,
    }
    write_json(output / "progress.json", status)
    print(json.dumps(status), flush=True)

  with dataset_lock(output):
    try:
      progress("collecting")
      for lead in plan["candidates"]:
        require_dataset_active(output)
        if lead["speciesId"] not in allowed or lead["split"] != "train":
          raise ValueError("Candidate outside approved species/split.")
        key = (lead["speciesId"], lead["observerGroup"])
        completed += 1
        if lead["photoId"] in seen_photos or lead["observationId"] in seen_observations:
          rejected["already-present"] += 1
          continue
        if author_counts[key] >= 3:
          rejected["author-cap"] += 1
          continue
        reserve_space(output, 32 * 1024**2)
        temporary = output / "images/current.download"
        try:
          payload = client.json(f"observations/{lead['observationId']}", {})
          row = refresh_candidate(payload, lead)
          if row is None:
            rejected["current-source-identity-license-or-quality"] += 1
            continue
          temporary.write_bytes(client.get(row["imageUrl"], limit=4 * 1024**2))
          with read_photo(temporary) as image:
            if min(image.size) < 100:
              rejected["too-small"] += 1
              continue
            pre_hash = difference_hash(image)
            image.save(temporary, format="JPEG", quality=92)
          with read_photo(temporary) as encoded:
            post_hash = difference_hash(encoded)
          digest = sha256(temporary)
          if (
            digest in frozen_sha or index.contains(digest, pre_hash)
            or index.contains(digest, post_hash)
          ):
            rejected["exact-or-near-duplicate"] += 1
            continue
          relative = f"images/{row['photoId']}.jpg"
          destination = output / relative
          if destination.exists():
            raise ValueError("Refuse overwriting downloaded image.")
          temporary.replace(destination)
          sample = {
            **row, "path": relative, "imageSha256": digest,
            "differenceHash": f"{post_hash:016x}",
            "preEncodingDifferenceHash": f"{pre_hash:016x}",
            "sourceQualityGrade": "research", "sourceCheckedAt": now(),
            "automatedSourceChecksPassed": True, "trainingEligible": False,
            "admissionStatus": "pending-consolidated-dataset-audit",
            "modifications": "RGB JPEG re-encode; metadata stripped; fixed source medium image",
          }
          write_json(output / "samples" / f"{row['photoId']}.json", {"samples": [sample]})
          rows.append(sample)
          index.add(digest, pre_hash)
          index.add(digest, post_hash)
          seen_photos.add(row["photoId"])
          seen_observations.add(row["observationId"])
          author_counts[key] += 1
        except requests.HTTPError as exc:
          if exc.response.status_code not in {404, 410}:
            raise
          rejected["source-removed"] += 1
        except ValueError as exc:
          # Decode failures are bounded data errors, but do not hide identity/code bugs.
          if str(exc) not in {
            "Photo cannot be decoded safely.", "Only JPEG, PNG and WebP are supported.",
            "Animated photos are not supported.", "Photo exceeds the 20-megapixel limit.",
          }:
            raise
          rejected["invalid-image"] += 1
        finally:
          temporary.unlink(missing_ok=True)
          progress("collecting")
      progress("collection-pass-complete")
    except BaseException as exc:
      write_json(output / "stopped.json", {
        "updatedAt": now(), "reasonType": type(exc).__name__,
        "automaticRetry": False, "requiresExplicitResumeReview": True,
      })
      progress("stopped")
      raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=["prepare", "collect"])
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--repo", type=Path, default=Path.cwd())
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.action == "prepare":
    plan = prepare(args.repo.resolve(), args.plan)
    print(json.dumps({"candidates": len(plan["candidates"]), "classes": len(plan["classes"])}))
  elif args.output:
    collect(args.plan, args.output)
  else:
    parser.error("--output required for collect")


if __name__ == "__main__":
  main()
