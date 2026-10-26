"""Prepare a bounded binary dataset from explicit Open Images human labels."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from yizhu_server.plantnet_pretrained import sha256, write_report
from yizhu_server.recognition import read_photo
from yizhu_server.training_data import (
  DuplicateIndex,
  dataset_lock,
  difference_hash,
  group_split,
  require_dataset_active,
)

PLANT = "/m/05s2s"
HUMAN_SOURCES = {"verification", "crowdsource-verification"}
LABEL_FILE = "oidv7-val-annotations-human-imagelabels.csv"
IMAGE_FILE = "validation-images-with-rotation.csv"
CLASS_FILE = "oidv7-class-descriptions.csv"
HIERARCHY_FILE = "bbox_labels_600_hierarchy.json"
LICENSE = "https://creativecommons.org/licenses/by/2.0/"
TARGETS = {"train": 200, "validation": 50, "test": 50}  # Per class.
_HTTP = threading.local()


def walk_hierarchy(node):
  yield node
  for key in ("Subcategory", "Part"):
    for child in node.get(key, []):
      yield from walk_hierarchy(child)


def explicit_labels(records, plant_related):
  """Missing labels and machine predictions are never negative ground truth."""
  states, sources, related_positive = defaultdict(set), defaultdict(set), set()
  for row in records:
    if row["Source"] not in HUMAN_SOURCES:
      continue
    value = float(row["Confidence"])
    if value not in (0, 1):
      continue
    if row["LabelName"] == PLANT:
      states[row["ImageID"]].add(int(value))
      sources[row["ImageID"]].add(row["Source"])
    if row["LabelName"] in plant_related and value == 1:
      related_positive.add(row["ImageID"])
  labels = {}
  for image_id, values in states.items():
    if values == {1} or values == {0} and image_id not in related_positive:
      labels[image_id] = {
        "label": "plant" if values == {1} else "nonplant",
        "confidence": next(iter(values)), "sources": sorted(sources[image_id]),
      }
  return labels


def select(root, targets=None, *, previous=None):
  targets = TARGETS if targets is None else targets
  if set(targets) != set(TARGETS) or any(
    type(value) is not int or value < 1 for value in targets.values()
  ):
    raise ValueError("Require positive per-class targets for train/validation/test.")
  meta = root / "metadata"
  with (meta / CLASS_FILE).open() as stream:
    names = dict(csv.reader(stream))
  if names.get(PLANT) != "Plant":
    raise ValueError("Official Plant class identity mismatch.")
  hierarchy = json.loads((meta / HIERARCHY_FILE).read_text())
  plant_node = next(node for node in walk_hierarchy(hierarchy) if node["LabelName"] == PLANT)
  # Conservative conflict removal, not inferred negative labels for absent tags.
  related = {node["LabelName"] for node in walk_hierarchy(plant_node)}
  related.update(key for key, name in names.items()
                 if name in {"Grass", "Leaf", "Moss", "Algae", "Fern"})
  with (meta / LABEL_FILE).open() as stream:
    labels = explicit_labels(csv.DictReader(stream), related)
  with (meta / IMAGE_FILE).open() as stream:
    images = list(csv.DictReader(stream))
  old = json.loads((previous / "samples.json").read_text()) if previous else None
  protected = [row for row in old["samples"] if row["split"] == "test"] if old else []
  protected_groups = {row["group"] for row in protected}
  protected_ids = {row["id"].rsplit(":", 1)[-1] for row in protected}
  protected_md5 = {row["OriginalMD5"] for row in images if row["ImageID"] in protected_ids}
  images.sort(key=lambda row: hashlib.sha256(
    f"presence-v1:{row['ImageID']}".encode(),
  ).hexdigest())
  pins = {filename: sha256(meta / filename)
          for filename in (LABEL_FILE, IMAGE_FILE, CLASS_FILE, HIERARCHY_FILE)}
  selected, counts, used_groups, used_md5 = [], Counter(), set(), set()
  for image in images:
    identifier = image["ImageID"]
    if (
      identifier not in labels or image["License"] != LICENSE
      or not re.fullmatch(r"[a-f0-9]{16}", identifier)
      or not image["AuthorProfileURL"].strip() or not image["Author"].strip()
      or not image["OriginalMD5"].strip()
      or urlsplit(image["OriginalLandingURL"]).scheme != "https"
    ):
      continue
    group = hashlib.sha256(
      f"openimages-author:{image['AuthorProfileURL'].rstrip('/')}".encode(),
    ).hexdigest()
    split, label = group_split(group), labels[identifier]["label"]
    if (
      group in protected_groups or image["OriginalMD5"] in protected_md5
      or group in used_groups or image["OriginalMD5"] in used_md5
      or counts[split, label] >= targets[split]
    ):
      continue
    selected.append({
      "id": f"openimages:validation:{identifier}", "path": f"images/{identifier}.jpg",
      "label": label, "split": split, "group": group,
      "license": "cc-by", "licenseUrl": LICENSE, "attribution": image["Author"],
      "sourceUrl": image["OriginalLandingURL"],
      "imageUrl": f"https://open-images-dataset.s3.amazonaws.com/validation/{identifier}.jpg",
      "public": True, "reviewed": True,
      "reviewMethod": "dataset-human-verified-image-level-label",
      "labelEvidence": {
        "dataset": "Open Images V7", "sourceSubset": "validation",
        "classId": PLANT, "confidence": labels[identifier]["confidence"],
        "sources": labels[identifier]["sources"], "annotationsSha256": pins[LABEL_FILE],
      },
    })
    counts[split, label] += 1
    used_groups.add(group)
    used_md5.add(image["OriginalMD5"])
  if any(counts[split, label] != target
         for split, target in targets.items() for label in ("plant", "nonplant")):
    raise ValueError("Not enough licensed independent authors for requested splits.")
  return {
    "version": 1, "source": "Open Images V7 human-verified image-level labels",
    "sourceSubset": "validation",
    "splitPolicy": "project-specific author-hash split; NOT official benchmark splits",
    "metadataSha256": pins, "targetsPerClass": targets,
    "eligibleExplicitLabels": dict(Counter(row["label"] for row in labels.values())),
    "negativeConflictGuardClasses": {key: names[key] for key in sorted(related)},
    "previousDataset": ({
      "path": str(previous.resolve()), "sha256": sha256(previous / "samples.json"),
      "protectedTestIds": sorted(row["id"] for row in protected),
    } if previous else None),
    "samples": selected,
  }


def download_one(root, row):
  import requests

  if not hasattr(_HTTP, "session"):
    _HTTP.session = requests.Session()
  path = root / row["path"]
  raw_digest = None
  if not path.exists():
    if shutil.disk_usage(root).free < 4 * 1024**3:
      raise OSError("Keep at least 4 GiB free; collection paused.")
    temporary = path.with_suffix(".download")
    for attempt in range(3):
      try:
        with _HTTP.session.get(
          row["imageUrl"], stream=True, timeout=(30, 90), allow_redirects=False,
        ) as response:
          if response.status_code in (403, 429):
            raise PermissionError("Source denied/rate-limited request; stop collection.")
          if response.is_redirect:
            raise ValueError("Unexpected image redirect.")
          response.raise_for_status()
          total = 0
          with temporary.open("wb") as stream:
            for chunk in response.iter_content(65536):
              total += len(chunk)
              if total > 8 * 1024 * 1024:
                raise ValueError("Image exceeds 8 MiB.")
              stream.write(chunk)
        raw_digest = sha256(temporary)
        with read_photo(temporary) as image:
          image.save(temporary, format="JPEG", quality=92)
        temporary.replace(path)
        break
      except (requests.ConnectionError, requests.Timeout):
        if attempt == 2:
          raise
        _HTTP.session.close()
        _HTTP.session = requests.Session()
        time.sleep(2 * (attempt + 1))
      finally:
        temporary.unlink(missing_ok=True)
  with read_photo(path) as image:
    perceptual = difference_hash(image)
  return {
    **row, "sha256": sha256(path), "differenceHash": f"{perceptual:016x}",
    "downloadedFileSha256": raw_digest,
    "modifications": "Official S3 image; EXIF orientation applied; RGB JPEG; metadata stripped",
  }


def download_sample(root, row):
  import requests

  try:
    return download_one(root, row)
  except ValueError as exc:
    # Image safety/format restrictions are decided before any model is trained.
    # Keep exclusions in the report, rather than silently loosening upload limits.
    return {"id": row["id"], "excludedReason": str(exc)}
  except requests.HTTPError as exc:
    if exc.response.status_code not in (404, 410):
      raise
    return {"id": row["id"], "excludedReason": "public-image-no-longer-available"}


def collect(root, *, workers=4, targets=None, previous=None, progress=None):
  require_dataset_active(root)
  if not 1 <= workers <= 4 or (root / "samples.json").exists():
    raise ValueError("Use 1–4 workers and a dataset without a frozen samples.json.")
  with dataset_lock(root):
    selection_path = root / "selection.json"
    if not selection_path.exists():
      write_report(selection_path, select(root, targets, previous=previous))
    selection = json.loads(selection_path.read_text())
    if targets is not None and selection["targetsPerClass"] != targets:
      raise ValueError("Frozen selection has different targets.")
    (root / "images").mkdir(exist_ok=True)
    rows, dropped, duplicates = [], [], DuplicateIndex()
    reference = selection.get("previousDataset")
    if reference:
      source = Path(reference["path"])
      if sha256(source / "samples.json") != reference["sha256"]:
        raise ValueError("Previous dataset changed.")
      old_rows = json.loads((source / "samples.json").read_text())["samples"]
      by_id = {row["id"]: row for row in old_rows}
      for row in old_rows:
        if row["split"] == "test":
          duplicates.add(row["sha256"], int(row["differenceHash"], 16))
      # Reuse only unchanged public training/validation files, never old test photos.
      for row in selection["samples"]:
        old_row = by_id.get(row["id"])
        if old_row is None:
          continue
        if old_row["split"] == "test" or old_row["split"] != row["split"]:
          raise ValueError("Previous split contamination.")
        old_path, new_path = source / old_row["path"], root / row["path"]
        if sha256(old_path) != old_row["sha256"]:
          raise ValueError("Cached public photo changed.")
        if not new_path.exists():
          shutil.copyfile(old_path, new_path)
    stopped = threading.Event()

    def fetch(item):
      if stopped.is_set():
        raise RuntimeError("Collection stopped after an earlier download failure.")
      try:
        return download_sample(root, item)
      except Exception:
        stopped.set()
        raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
      for index, row in enumerate(
        pool.map(fetch, selection["samples"]), 1,
      ):
        if "excludedReason" in row:
          dropped.append({"id": row["id"], "reason": row["excludedReason"]})
          continue
        perceptual = int(row["differenceHash"], 16)
        if duplicates.contains(row["sha256"], perceptual):
          dropped.append({"id": row["id"], "reason": "exact-or-near-duplicate"})
        else:
          duplicates.add(row["sha256"], perceptual)
          rows.append(row)
        if index % 25 == 0:
          print(f"Downloaded {index}/{len(selection['samples'])}", flush=True)
          if progress:
            progress({
              "processedImages": index, "totalImages": len(selection["samples"]),
              "acceptedImages": len(rows), "excludedImages": len(dropped),
            })
    report = {**selection, "samples": rows, "excludedBeforeTraining": dropped}
    write_report(root / "samples.json", report)
    print(json.dumps({
      "samples": len(rows), "excludedBeforeTraining": len(dropped),
      "counts": {split: dict(Counter(row["label"] for row in rows if row["split"] == split))
                 for split in TARGETS},
    }, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True, type=Path)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--train-per-class", type=int, default=200)
  parser.add_argument("--validation-per-class", type=int, default=50)
  parser.add_argument("--test-per-class", type=int, default=50)
  parser.add_argument("--previous", type=Path)
  args = parser.parse_args()
  collect(args.data, workers=args.workers, previous=args.previous, targets={
    "train": args.train_per_class, "validation": args.validation_per_class,
    "test": args.test_per_class,
  })


if __name__ == "__main__":
  main()
