"""Offline plant-presence classifier. No downloads, species lookup or API activation."""

import argparse
import hashlib
import json
import math
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from yizhu_server.plantnet_pretrained import (
  DEFAULT_DIRECTORY,
  PREPROCESS,
  PlantNetPretrained,
  sha256,
  write_report,
)
from yizhu_server.recognition import read_photo
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import DuplicateIndex, difference_hash, require_dataset_active

LABELS = ["nonplant", "plant"]
DIMENSION = 2048
SPLITS = ("train", "validation", "test")
PREPROCESS_ID = "whole-image-letterbox224-imagenet-normalization-l2-v1"
BACKBONES = ("plantnet", "imagenet")
IMAGENET_DIRECTORY = DEFAULT_DIRECTORY.parent / "imagenet-resnet50-v2"
IMAGENET_FILENAME = "resnet50-11ad3fa6.pth"
IMAGENET_BYTES = 102540417
IMAGENET_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"


def load_dataset(root, *, required_splits=SPLITS):
  """Require reviewed whole-image labels, including dataset human verification."""
  required_splits = tuple(required_splits)
  if required_splits not in (SPLITS, ("train", "validation")):
    raise ValueError("Require all splits or explicitly train/validation only.")
  path = root / "samples.json"
  if not 0 < path.stat().st_size <= 16 * 1024 * 1024:
    raise ValueError("Invalid dataset manifest size.")
  document = json.loads(path.read_text())
  if not isinstance(document, dict) or document.get("version") != 1:
    raise ValueError("Expected dataset format version 1.")
  rows = document.get("samples")
  if not isinstance(rows, list) or not 2 * len(required_splits) <= len(rows) <= 10000:
    raise ValueError("Need reviewed positive and negative photos in every required split.")
  ids, groups, counts, duplicates = set(), {}, Counter(), DuplicateIndex()
  for row in rows:
    if not isinstance(row, dict) or any(
      not isinstance(row.get(key), str) or not row[key].strip()
      for key in ("id", "path", "sha256", "label", "split", "group",
                  "license", "attribution", "sourceUrl")
    ):
      raise ValueError("Incomplete sample metadata.")
    if (
      row["label"] not in LABELS or row["split"] not in required_splits
      or row["license"] not in {"cc0", "cc-by"}
      or row.get("public") is not True or row.get("reviewed") is not True
      or urlsplit(row["sourceUrl"]).scheme != "https"
      or not urlsplit(row["sourceUrl"]).netloc
    ):
      raise ValueError("Require public licensed photos and reviewed whole-image binary labels.")
    evidence = row.get("labelEvidence")
    if evidence is not None and (
      not isinstance(evidence, dict)
      or evidence.get("dataset") != "Open Images V7" or evidence.get("classId") != "/m/05s2s"
      or evidence.get("confidence") != LABELS.index(row["label"])
      or not evidence.get("sources")
      or not set(evidence["sources"]) <= {"verification", "crowdsource-verification"}
    ):
      raise ValueError("Binary label contradicts explicit dataset human verification.")
    relative = Path(row["path"])
    photo = (root / relative).resolve()
    if (
      relative.is_absolute() or ".." in relative.parts
      or relative.parts[0] != "images"
      or not photo.is_relative_to((root / "images").resolve())
      or not photo.is_relative_to(root.resolve())
    ):
      raise ValueError("Image path escapes dataset.")
    if sha256(photo) != row["sha256"]:
      raise ValueError("Image checksum mismatch.")
    if row["id"] in ids:
      raise ValueError("Duplicate sample ID.")
    ids.add(row["id"])
    previous = groups.setdefault(row["group"], row["split"])
    if previous != row["split"]:
      raise ValueError("Related photographer/scene/observation group leaks across splits.")
    with read_photo(photo) as image:
      perceptual = difference_hash(image)
    if duplicates.contains(row["sha256"], perceptual):
      raise ValueError("Exact or near-duplicate photos; review dataset before training.")
    duplicates.add(row["sha256"], perceptual)
    counts[row["split"], row["label"]] += 1
  if any(not counts[split, label] for split in required_splits for label in LABELS):
    raise ValueError("Each split must contain both plant and nonplant photos.")
  return rows, sha256(path)


def whole_image(image):
  """Preserve edge plants instead of throwing them away with a center crop."""
  from PIL import Image, ImageOps

  resized = ImageOps.contain(image.convert("RGB"), (224, 224), Image.Resampling.BILINEAR)
  canvas = Image.new("RGB", (224, 224), (124, 116, 104))
  canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
  return canvas


def imagenet_model(directory):
  """Strict, offline loading of the downloaded official IMAGENET1K_V2 weights."""
  import torch
  from torchvision import models

  path = directory / IMAGENET_FILENAME
  if (
    not path.is_file() or path.stat().st_size != IMAGENET_BYTES
    or sha256(path) != IMAGENET_SHA256
  ):
    raise ValueError("Pinned ImageNet ResNet50 V2 checkpoint missing or changed.")
  state = torch.load(path, map_location="cpu", weights_only=True)
  if (
    not isinstance(state, dict) or "fc.weight" not in state
    or not isinstance(state["fc.weight"], torch.Tensor)
    or tuple(state["fc.weight"].shape) != (1000, DIMENSION)
  ):
    raise ValueError("Unexpected ImageNet checkpoint classifier.")
  model = models.resnet50(weights=None, num_classes=1000)
  model.load_state_dict(state, strict=True)
  return model.eval().requires_grad_(False)


class PresenceEncoder:
  def __init__(self, assets=None, *, backbone="plantnet", threads=1):
    import torch
    from torchvision import transforms

    if backbone not in BACKBONES or type(threads) is not int or not 1 <= threads <= 4:
      raise ValueError("Use a registered backbone and 1–4 CPU threads.")
    torch.set_num_threads(threads)
    if backbone == "plantnet":
      pretrained = PlantNetPretrained(
        assets or DEFAULT_DIRECTORY, architecture="resnet50", threads=threads,
      )
      self.model = pretrained.model
      digest = pretrained.manifest["files"]["resnet50_weights_best_acc.tar"]["sha256"]
    else:
      self.model = imagenet_model(assets or IMAGENET_DIRECTORY)
      digest = IMAGENET_SHA256
    self.identity = {
      "architecture": "resnet50", "weightsSha256": digest,
      "dimension": DIMENSION, "preprocessing": PREPROCESS_ID,
    }
    self.model.fc = torch.nn.Identity()  # Discard the original classification head.
    self.preprocess = transforms.Compose([
      transforms.ToTensor(), transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])

  def encode(self, image):
    import torch

    with whole_image(image) as full, torch.inference_mode(), torch.backends.nnpack.flags(
      enabled=False,
    ):
      vector = self.model(self.preprocess(full).unsqueeze(0))
      if vector.shape != (1, DIMENSION) or not torch.isfinite(vector).all():
        raise ValueError("Invalid encoder output.")
      if float(vector.norm()) <= 0:
        raise ValueError("Zero encoder feature.")
      return torch.nn.functional.normalize(vector, dim=1)[0]


def binary_metrics(scores, labels, threshold):
  if (
    not math.isfinite(threshold) or not 0 <= threshold <= 1
    or not scores or len(scores) != len(labels)
    or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
    or any(label not in (0, 1) for label in labels)
  ):
    raise ValueError("Invalid binary scores, labels or threshold.")
  tp = sum(score >= threshold and label == 1 for score, label in zip(scores, labels))
  fp = sum(score >= threshold and label == 0 for score, label in zip(scores, labels))
  tn = sum(score < threshold and label == 0 for score, label in zip(scores, labels))
  fn = sum(score < threshold and label == 1 for score, label in zip(scores, labels))
  return {
    "samples": len(labels), "truePositive": tp, "falsePositive": fp,
    "trueNegative": tn, "falseNegative": fn,
    "plantRecall": tp / (tp + fn) if tp + fn else None,
    "nonplantRecall": tn / (tn + fp) if tn + fp else None,
    "plantPrecision": tp / (tp + fp) if tp + fp else None,
    "falsePositiveRate": fp / (tn + fp) if tn + fp else None,
    "accuracy": (tp + tn) / len(labels),
    "balancedAccuracy": ((tp / (tp + fn) + tn / (tn + fp)) / 2
                         if tp + fn and tn + fp else None),
  }


def select_threshold(scores, labels, minimum_recall):
  """Highest threshold retaining the requested empirical VALIDATION plant recall."""
  binary_metrics(scores, labels, 0.5)
  if set(labels) != {0, 1} or not 0 < minimum_recall <= 1:
    raise ValueError("Threshold selection needs both classes and a valid recall target.")
  positives = sorted((score for score, label in zip(scores, labels) if label), reverse=True)
  return positives[math.ceil(len(positives) * minimum_recall) - 1]


def cached_feature(encoder, root, row):
  import numpy as np
  import torch

  identity = {**encoder.identity, "torch": version("torch"), "torchvision": version("torchvision")}
  namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
  folder = root / "features" / namespace
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / f"{row['sha256']}.npy"
  if not path.exists():
    reserve_space(folder, DIMENSION * 4 + 1024)
    with read_photo(root / row["path"]) as image:
      array = encoder.encode(image).cpu().numpy().astype(np.float32)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
      np.save(stream, array, allow_pickle=False)
    temporary.replace(path)
  if not 0 < path.stat().st_size < DIMENSION * 4 + 1024:
    raise ValueError("Invalid feature cache size.")
  array = np.load(path, allow_pickle=False)
  if (
    array.shape != (DIMENSION,) or array.dtype != np.float32
    or not np.isfinite(array).all() or not np.isclose(np.linalg.norm(array), 1, atol=0.001)
  ):
    raise ValueError("Invalid normalized feature cache.")
  return torch.from_numpy(array)


def encode_rows(encoder, root, rows, *, progress=None):
  import torch

  vectors = []
  for index, row in enumerate(rows, 1):
    vectors.append(cached_feature(encoder, root, row))
    if index % 25 == 0 or index == len(rows):
      print(f"Encoded {index}/{len(rows)} {rows[0]['split']} images", flush=True)
      if progress:
        progress({"phase": "encoding", "split": rows[0]["split"],
                  "processedImages": index, "totalImages": len(rows)})
  return torch.stack(vectors), torch.tensor([LABELS.index(row["label"]) for row in rows])


def train(root, output, *, assets=None, epochs=100, minimum_recall=0.95,
          backbone="plantnet", threads=1, progress=None, head_type="linear"):
  require_dataset_active(root)
  if output.exists() or not 1 <= epochs <= 1000 or not 0 < minimum_recall <= 1:
    raise ValueError("Use a new output file and valid epochs/recall target.")
  if head_type not in {"linear", "mlp32"}:
    raise ValueError("Unknown binary head type.")
  import torch

  from yizhu_server.training_classifier import fit_head

  rows, digest = load_dataset(root)
  encoder = PresenceEncoder(assets, backbone=backbone, threads=threads)
  # Test images are validated above but never encoded/scored to select weights or threshold.
  splits = {
    split: encode_rows(
      encoder, root, [row for row in rows if row["split"] == split], progress=progress,
    )
    for split in ("train", "validation")
  }
  identity = encoder.identity
  del encoder
  if progress:
    progress({"phase": "fitting-head"})
  head, history = fit_head(
    splits, 2, epochs=epochs, hidden_dimension=32 if head_type == "mlp32" else 0,
  )
  with torch.inference_mode():
    scores = head(splits["validation"][0]).softmax(1)[:, 1].tolist()
  labels = splits["validation"][1].tolist()
  threshold = select_threshold(scores, labels, minimum_recall)
  report = {
    "format": 2 if head_type == "mlp32" else 1,
    "task": "plant-presence", "labels": LABELS, "backbone": backbone,
    "encoder": identity, "threshold": threshold,
    **({
      "headType": head_type,
      "state": {name: tensor.detach().tolist() for name, tensor in head.state_dict().items()},
    } if head_type == "mlp32" else {
      "weight": head.weight.detach().tolist(), "bias": head.bias.detach().tolist(),
    }),
    "datasetSha256": digest, "minimumValidationPlantRecall": minimum_recall,
    "validation": binary_metrics(scores, labels, threshold),
    "counts": {split: dict(Counter(
      row["label"] for row in rows if row["split"] == split
    )) for split in SPLITS},
    "training": history, "confidenceCalibrated": False, "productionApproved": False,
    "testUsedForSelection": False,
  }
  write_report(output, report)
  from yizhu_server.solution_metadata import write_artifact_metadata

  write_artifact_metadata(output)
  return report


class PlantPresenceClassifier:
  """A real trained head is mandatory; no random-head or species-score fallback."""

  def __init__(self, head_path, *, assets=None, threads=1):
    import torch

    if not 0 < head_path.stat().st_size <= 8 * 1024 * 1024:
      raise ValueError("Invalid head size.")
    report = json.loads(head_path.read_text())
    if (
      not isinstance(report, dict) or report.get("format") not in {1, 2}
      or report.get("task") != "plant-presence" or report.get("labels") != LABELS
      or report.get("confidenceCalibrated") is not False
      or not isinstance(report.get("threshold"), (int, float))
      or isinstance(report["threshold"], bool)
      or not math.isfinite(report["threshold"]) or not 0 <= report["threshold"] <= 1
    ):
      raise ValueError("Invalid binary head metadata.")
    if report["format"] == 2 and report.get("headType") != "mlp32":
      raise ValueError("Unknown binary head architecture.")
    mlp = report["format"] == 2
    expected = (
      {"0.weight": (32, DIMENSION), "0.bias": (32,), "2.weight": (2, 32), "2.bias": (2,)}
      if mlp else {"weight": (2, DIMENSION), "bias": (2,)}
    )
    try:
      raw = report["state"] if mlp else {"weight": report["weight"], "bias": report["bias"]}
      if not isinstance(raw, dict) or set(raw) != set(expected):
        raise ValueError("Unexpected state keys.")
      state = {name: torch.tensor(value, dtype=torch.float32) for name, value in raw.items()}
    except (KeyError, ValueError, TypeError, RuntimeError):
      raise ValueError("Invalid binary head weights.") from None
    if any(tensor.shape != expected[name] or not torch.isfinite(tensor).all()
           for name, tensor in state.items()):
      raise ValueError("Invalid binary head dimensions/values.")
    self.encoder = PresenceEncoder(
      assets, backbone=report.get("backbone", "plantnet"), threads=threads,
    )
    if report.get("encoder") != self.encoder.identity:
      raise ValueError("Binary head encoder/preprocessing mismatch.")
    self.head = (
      torch.nn.Sequential(torch.nn.Linear(DIMENSION, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
      if mlp else torch.nn.Linear(DIMENSION, 2)
    )
    self.head.load_state_dict(state, strict=True)
    self.head.eval().requires_grad_(False)
    self.report = report
    self.version = sha256(head_path)

  def predict(self, image):
    return self.predict_vector(self.encoder.encode(image))

  def predict_vector(self, vector):
    import torch

    with torch.inference_mode():
      score = float(self.head(vector).softmax(0)[1])
    if not math.isfinite(score):
      raise ValueError("Invalid binary prediction.")
    return {
      "containsPlant": score >= self.report["threshold"],
      "score": score, "threshold": self.report["threshold"], "confidence": None,
      "scoreType": "binary_softmax_not_calibrated_confidence", "modelVersion": self.version,
      "productionApproved": False,
    }


def evaluate(root, head_path, output, *, assets=None, threads=1, progress=None):
  if output.exists():
    raise ValueError("Evaluation output exists; use a new path.")
  rows, digest = load_dataset(root)
  classifier = PlantPresenceClassifier(head_path, assets=assets, threads=threads)
  if digest != classifier.report.get("datasetSha256"):
    raise ValueError("Dataset changed since training; freeze the labeled splits.")
  scores, labels, cases = [], [], []
  for row in rows:
    if row["split"] != "test":
      continue
    result = classifier.predict_vector(cached_feature(classifier.encoder, root, row))
    scores.append(result["score"])
    labels.append(LABELS.index(row["label"]))
    cases.append({"id": row["id"], "expected": row["label"], **result})
    if len(cases) % 100 == 0:
      print(f"Evaluated {len(cases)} held-out test images", flush=True)
      if progress:
        progress({"processedImages": len(cases),
                  "totalImages": sum(row["split"] == "test" for row in rows)})
  report = {
    "task": "plant-presence", "split": "test", "modelVersion": classifier.version,
    "datasetSha256": digest, "threshold": classifier.report["threshold"],
    "metrics": binary_metrics(scores, labels, classifier.report["threshold"]),
    "cases": cases, "productionApproved": False,
  }
  write_report(output, report)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=("check-data", "train", "evaluate", "predict"))
  parser.add_argument("--data", type=Path)
  parser.add_argument("--assets", type=Path)
  parser.add_argument("--backbone", choices=BACKBONES, default="plantnet",
                      help="Training initialization; prediction reads it from the trained head")
  parser.add_argument("--threads", type=int, choices=range(1, 5), default=1)
  parser.add_argument("--head", type=Path)
  parser.add_argument("--photo", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--minimum-recall", type=float, default=0.95)
  parser.add_argument("--head-type", choices=("linear", "mlp32"), default="linear")
  args = parser.parse_args()
  needed = {
    "check-data": ("data",), "train": ("data", "output"),
    "evaluate": ("data", "head", "output"), "predict": ("head", "photo"),
  }
  if any(getattr(args, key) is None for key in needed[args.action]):
    parser.error(f"{args.action} requires: {', '.join(needed[args.action])}")
  if args.action == "check-data":
    rows, digest = load_dataset(args.data)
    result = {"samples": len(rows), "datasetSha256": digest}
  elif args.action == "train":
    report = train(args.data, args.output, assets=args.assets, epochs=args.epochs,
                   minimum_recall=args.minimum_recall, backbone=args.backbone,
                   threads=args.threads, head_type=args.head_type)
    result = {"validation": report["validation"], "productionApproved": False}
  elif args.action == "evaluate":
    result = evaluate(
      args.data, args.head, args.output, assets=args.assets, threads=args.threads,
    )["metrics"]
  else:
    classifier = PlantPresenceClassifier(args.head, assets=args.assets, threads=args.threads)
    with read_photo(args.photo) as image:
      result = classifier.predict(image)
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
