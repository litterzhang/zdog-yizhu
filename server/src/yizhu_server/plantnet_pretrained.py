"""Offline trials of official Pl@ntNet-300K checkpoints; no API or training."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import resource
import statistics
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache/models/plantnet-300k-pretrained"
CLASS_COUNT = 1081
ARCHITECTURES = {
  "resnet50": ("resnet50_weights_best_acc.tar", 206149183, "fc.weight", (1081, 2048)),
  "squeezenet1_0": (
    "squeezenet_weights_best_acc.tar", 10354289, "classifier.1.weight", (1081, 512, 1, 1),
  ),
}
MAP_FILES = {
  "class_idx_to_species_id.json": 24837,
  "plantnet300K_species_id_2_name.json": 46964,
}
PREPROCESS = {
  "resizeShortEdge": 256, "centerCrop": 224, "interpolation": "bilinear",
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}


def sha256(path):
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(data, stream, ensure_ascii=False, indent=2)
    stream.write("\n")


def read_class_map(directory):
  if any(not 0 < (directory / filename).stat().st_size <= 512 * 1024 for filename in MAP_FILES):
    raise ValueError("Unexpected class-map file size.")
  indices = json.loads((directory / "class_idx_to_species_id.json").read_text())
  names = json.loads((directory / "plantnet300K_species_id_2_name.json").read_text())
  if not isinstance(indices, dict) or set(indices) != {str(i) for i in range(CLASS_COUNT)}:
    raise ValueError("Expected the original 0–1080 index map, not V2's 1000-class metadata.")
  ids = [str(indices[str(i)]) for i in range(CLASS_COUNT)]
  if not isinstance(names, dict) or len(set(ids)) != CLASS_COUNT or set(ids) != set(names):
    raise ValueError("Official class-index/species-name maps do not agree.")
  if any(not isinstance(names[key], str) or not names[key].strip() for key in ids):
    raise ValueError("Empty source scientific name.")
  return [(source_id, names[source_id]) for source_id in ids]


def prepare(directory, architecture):
  """Pin downloaded files locally; this is not a publisher-provided checksum."""
  filename, size, _, _ = ARCHITECTURES[architecture]
  destination = directory / f"{architecture}.manifest.json"
  if destination.exists():
    raise ValueError("Manifest exists; use a new asset directory for a new version.")
  read_class_map(directory)
  files = {}
  for name, expected in {**MAP_FILES, filename: size}.items():
    path = directory / name
    if not path.is_file() or path.stat().st_size != expected:
      raise ValueError(f"Missing/incomplete official asset: {name}")
    files[name] = {"bytes": expected, "sha256": sha256(path)}
  manifest = {
    "format": 1, "architecture": architecture, "classCount": CLASS_COUNT,
    "datasetVariant": "official-1081-class-pretrained-not-V2",
    "source": "https://github.com/plantnet/PlantNet-300K",
    "preprocess": PREPROCESS, "files": files,
    "integrity": "Published directory sizes checked; local SHA-256, not publisher-signed hashes",
    "licenseStatus": "External checkpoint license not separately verified; local trial, not deployment",
  }
  write_report(destination, manifest)
  return manifest


def verify(directory, architecture):
  path = directory / f"{architecture}.manifest.json"
  if not path.is_file() or path.stat().st_size > 32768:
    raise ValueError("Prepare a local manifest before inference.")
  manifest = json.loads(path.read_text())
  filename, size, _, _ = ARCHITECTURES[architecture]
  if (
    manifest.get("format") != 1 or manifest.get("architecture") != architecture
    or manifest.get("classCount") != CLASS_COUNT
    or manifest.get("datasetVariant") != "official-1081-class-pretrained-not-V2"
    or manifest.get("preprocess") != PREPROCESS
    or set(manifest.get("files", {})) != {*MAP_FILES, filename}
  ):
    raise ValueError("Unexpected model manifest.")
  for name, expected in {**MAP_FILES, filename: size}.items():
    info = manifest["files"][name]
    path = directory / name
    if (
      not path.is_file() or path.stat().st_size != expected or info.get("bytes") != expected
      or sha256(path) != info.get("sha256")
    ):
      raise ValueError("Asset size/checksum mismatch.")
  return manifest


def class_records(class_map, catalog):
  from yizhu_server.recognition_plantclef import scientific_binomial

  names = [scientific_binomial(raw) for _, raw in class_map]
  counts = Counter(names)
  by_name = {}
  if catalog is not None and catalog.available:
    wanted = sorted({name for name in names if name is not None})
    with catalog.connect() as db:
      for start in range(0, len(wanted), 200):
        chunk = wanted[start:start + 200]
        for row in db.execute(
          f"SELECT scientific_name,data FROM species WHERE scientific_name IN "
          f"({','.join('?' for _ in chunk)})", chunk,
        ):
          by_name.setdefault(row["scientific_name"], []).append(json.loads(row["data"]))
  records = []
  for index, ((source_id, raw), name) in enumerate(zip(class_map, names, strict=True)):
    name = name if name and counts[name] == 1 else None
    matches = by_name.get(name, [])
    species = matches[0] if len(matches) == 1 else {}
    aliases = species.get("chineseAliases") or []
    records.append({
      "classIndex": index, "sourceSpeciesId": source_id, "sourceScientificName": raw,
      "scientificName": name, "catalogSpeciesId": species.get("id"),
      "chineseName": aliases[0] if aliases else species.get("chineseName"),
      "catalogVisible": species.get("isVisible", False),
    })
  return records


def top_candidates(records, scores, k=5):
  if (
    len(records) != len(scores) or not 1 <= k <= len(records)
    or any(isinstance(s, bool) or not isinstance(s, (int, float))
           or not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
  ):
    raise ValueError("Invalid classifier scores.")
  # Never skip an unmapped/hidden leader to manufacture a visible successful result.
  order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
  output, seen = [], set()
  for index in order:
    row = records[index]
    # Some official source IDs repeat an identical full scientific label.
    # Deduplicate those, without inventing equivalence between different authors.
    identity = row["scientificName"] or (
      "source-label:" + row.get("sourceScientificName", row["sourceSpeciesId"])
    )
    if identity in seen:
      continue
    seen.add(identity)
    output.append({**row, "score": float(scores[index]), "confidence": None})
    if len(output) == k:
      break
  return output


def summarize_cases(cases):
  groups = {}
  for row in cases:
    groups.setdefault(row["expectedScientificName"], []).append(row)
  n = len(cases)
  if not n:
    raise ValueError("No labeled evaluation photos.")
  times = sorted(row["seconds"] for row in cases if row["technicalSuccess"])
  return {
    "images": n, "targetSpecies": len(groups),
    "coveredSpecies": sum(any(row["labelCovered"] for row in rows) for rows in groups.values()),
    "top1Hits": sum(row["top1Hit"] for row in cases),
    "top5Hits": sum(row["top5Hit"] for row in cases),
    "top1": sum(row["top1Hit"] for row in cases) / n,
    "top5": sum(row["top5Hit"] for row in cases) / n,
    "macroTop5": statistics.mean(
      sum(row["top5Hit"] for row in rows) / len(rows) for rows in groups.values()
    ),
    "technicalSuccessRate": sum(row["technicalSuccess"] for row in cases) / n,
    "latencyMedianSeconds": statistics.median(times) if times else None,
    "latencyP95Seconds": times[max(0, math.ceil(len(times) * .95) - 1)] if times else None,
    "bySpecies": {
      name: {
        "images": len(rows), "labelCovered": any(row["labelCovered"] for row in rows),
        "top1Hits": sum(row["top1Hit"] for row in rows),
        "top5Hits": sum(row["top5Hit"] for row in rows),
      } for name, rows in sorted(groups.items())
    },
  }


class PlantNetPretrained:
  def __init__(self, directory=DEFAULT_DIRECTORY, *, architecture="resnet50", threads=1,
               catalog=None):
    if architecture not in ARCHITECTURES or not 1 <= threads <= 4:
      raise ValueError("Unsupported architecture or CPU thread count.")
    self.manifest = verify(directory, architecture)
    self.records = class_records(read_class_map(directory), catalog)
    import torch
    from torchvision import models, transforms
    from torchvision.transforms import InterpolationMode

    torch.set_num_threads(threads)
    filename, _, classifier_key, expected_shape = ARCHITECTURES[architecture]
    # No publisher Python execution, no unsafe pickle fallback, no hub calls.
    checkpoint = torch.load(directory / filename, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if (
      not isinstance(state, dict)
      or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
             for key, value in state.items())
      or classifier_key not in state or tuple(state[classifier_key].shape) != expected_shape
    ):
      raise ValueError("Unexpected checkpoint or output dimensions.")
    model = getattr(models, architecture)(weights=None, num_classes=CLASS_COUNT)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()
    model.eval()
    model.requires_grad_(False)
    self.model = model
    self.preprocess = transforms.Compose([
      transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
      transforms.CenterCrop(224), transforms.ToTensor(),
      transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])

  def predict(self, image):
    import torch

    # This CPU cannot initialize NNPACK; use the supported CPU convolution path.
    # Context restores the flag and does not alter the separate running server.
    with torch.inference_mode(), torch.backends.nnpack.flags(enabled=False):
      logits = self.model(self.preprocess(image).unsqueeze(0))
      if logits.shape != (1, CLASS_COUNT) or not torch.isfinite(logits).all():
        raise ValueError("Unexpected model output.")
      scores = logits.softmax(dim=1)[0].tolist()
    return top_candidates(self.records, scores)


def public_test_rows(data):
  """Reuse public cached test photos, not training splits or private user records."""
  from yizhu_server.training_data import load_samples

  rows = sorted((row for row in load_samples(data) if row["split"] == "test"),
                key=lambda row: row["sampleId"])
  if not rows or len(rows) > 500:
    raise ValueError("The bounded trial requires 1–500 cached public test photos.")
  if len({row["sampleId"] for row in rows}) != len(rows):
    raise ValueError("Duplicate evaluation sample IDs.")
  for row in rows:
    if (
      row.get("license") not in {"cc0", "cc-by"} or not row.get("attribution")
      or not row.get("imageUrl", "").startswith("https://inaturalist-open-data.s3.amazonaws.com/")
      or not row.get("sourceUrl", "").startswith("https://www.inaturalist.org/observations/")
    ):
      raise ValueError("Evaluation input lacks public photo provenance.")
    path = (data / row["path"]).resolve()
    if not path.is_relative_to((data / "images").resolve()) or sha256(path) != row["imageSha256"]:
      raise ValueError("Unsafe path or changed public test image.")
  return rows


def evaluate(provider, data):
  from yizhu_server.recognition import read_photo

  rows = public_test_rows(data)
  covered = {row["scientificName"] for row in provider.records if row["scientificName"]}
  cases = []
  for index, row in enumerate(rows, 1):
    before = time.monotonic()
    candidates, success = [], True
    try:
      with read_photo(data / row["path"]) as image:
        candidates = provider.predict(image)
    except (ValueError, OSError, RuntimeError):
      success = False
    names = [candidate["scientificName"] for candidate in candidates]
    expected = row["scientificName"]
    cases.append({
      "sampleId": row["sampleId"], "imageSha256": row["imageSha256"],
      "expectedScientificName": expected, "labelCovered": expected in covered,
      "technicalSuccess": success, "seconds": round(time.monotonic() - before, 4),
      "top1Hit": bool(names) and names[0] == expected, "top5Hit": expected in names,
      "candidates": candidates,
    })
    if index % 10 == 0 or index == len(rows):
      print(f"Evaluated {index}/{len(rows)} public test photos", flush=True)
  return {
    "purpose": "small local trial, not representative China-wide accuracy",
    "split": "cached-public-test-only", "metrics": summarize_cases(cases), "cases": cases,
    "testManifestSha256": hashlib.sha256(json.dumps([
      {key: row[key] for key in ("sampleId", "imageSha256", "scientificName", "split")}
      for row in rows
    ], sort_keys=True).encode()).hexdigest(),
    "limitations": [
      "Few species, geographically/observer-biased sample collection.",
      "Public community labels, not expert-audited gold labels.",
      "Foundation-training overlap not ruled out.",
      "Conservative exact binomial matching; unreviewed synonyms not bridged.",
      "Original full 1081-class output; no target-specific shortlist or retraining.",
      "Uncovered labels and inference errors remain in the denominator.",
      "Softmax scores are not calibrated confidence; closed set has no nonplant rejection.",
    ],
  }


def controls(provider):
  from PIL import Image

  images = [
    ("blank", Image.new("RGB", (224, 224), "white")),
    ("noise", Image.frombytes("RGB", (224, 224), random.Random(0).randbytes(224 * 224 * 3))),
  ]
  results = []
  for name, image in images:
    with image:
      results.append({"case": name, "candidates": provider.predict(image)})
  return {"purpose": "synthetic closed-set sanity checks; not a nonplant accuracy test",
          "results": results}


def main():
  os.environ["HF_HUB_OFFLINE"] = "1"
  os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
  os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=("prepare", "predict", "evaluate"))
  parser.add_argument("--assets", type=Path, default=DEFAULT_DIRECTORY)
  parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="resnet50")
  parser.add_argument("--photo", type=Path)
  parser.add_argument("--data", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--threads", type=int, choices=range(1, 5), default=1)
  parser.add_argument("--include-controls", action="store_true")
  args = parser.parse_args()
  if args.action == "prepare":
    manifest = prepare(args.assets, args.architecture)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return
  if args.output is None or args.output.exists():
    parser.error("Inference requires a new --output path.")
  if args.action == "predict" and args.photo is None:
    parser.error("predict requires --photo.")
  if args.action == "evaluate" and args.data is None:
    parser.error("evaluate requires --data with public cached test samples.")
  from yizhu_server.catalog import PlantCatalog
  from yizhu_server.recognition import read_photo

  started = time.monotonic()
  provider = PlantNetPretrained(
    args.assets, architecture=args.architecture, threads=args.threads, catalog=PlantCatalog(),
  )
  initialization = time.monotonic() - started
  if args.action == "predict":
    before = time.monotonic()
    with read_photo(args.photo) as image:
      report = {"purpose": "single-photo local trial", "candidates": provider.predict(image)}
    report["seconds"] = round(time.monotonic() - before, 4)
  else:
    report = evaluate(provider, args.data)
  if args.include_controls:
    report["controls"] = controls(provider)
  report.update({
    "architecture": args.architecture, "manifest": provider.manifest,
    "classCount": len(provider.records), "confidenceCalibrated": False,
    "scoreType": "softmax_score_not_calibrated_confidence",
    "productionChanged": False, "trainingPerformed": False, "device": "cpu",
    "threads": args.threads, "initializationSeconds": round(initialization, 4),
    "nnpackEnabledForTrial": False,
    "totalSeconds": round(time.monotonic() - started, 4),
    "peakProcessRssMiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    "versions": {name: version(name) for name in ("torch", "torchvision", "pillow")},
  })
  write_report(args.output, report)
  print(json.dumps(report.get("metrics", report.get("candidates")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
