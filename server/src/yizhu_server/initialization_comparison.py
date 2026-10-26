"""Fixed, offline ResNet18 initialization comparison; never reads a test split."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import now, require_dataset_active, sha256, write_json

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
MANIFEST_SHA = "0743762504446a0eb546fd8d1117901413559422c9ed3ae6402041715de8bbe3"
RECIPE = {
  "architecture": "resnet18", "classCount": 82, "epochs": 30, "patience": 10,
  "seed": 20260925, "headSeed": 2026092501, "confirmationSeed": 20260926,
  "confirmationHeadSeed": 2026092601, "automaticConfirmation": False,
  "batchSize": 16, "threads": 2, "imageSize": 224, "resizeShortEdge": 256,
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
  "interpolation": "bilinear", "antialias": True,
  "augmentation": {
    "randomResizedCropScale": [0.7, 1.0], "randomResizedCropRatio": [0.75, 4 / 3],
    "horizontalFlip": 0.5,
    "colorJitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.02},
  },
  "optimizer": "AdamW", "learningRate": 0.001, "weightDecay": 0.01,
  "betas": [0.9, 0.999], "eps": 1e-8, "gradientClipNorm": 5.0,
  "scheduler": "cosine", "etaMin": 1e-5, "trainBackbone": True,
  "selection": "strict lowest validation cross entropy; earliest epoch on exact tie",
  "engineeringThresholdTop5": 0.05,
  "bootstrap": {"unit": "observerGroup", "draws": 2000, "seed": 2026092502},
  "testUsed": False, "automaticDeployment": False,
}


def load_data(data, manifest_sha=MANIFEST_SHA):
  """Reuse the frozen raw data without changing its original scratch-only declaration."""
  require_dataset_active(data)
  manifest_path = data / "train-validation.json"
  if sha256(manifest_path) != manifest_sha:
    raise ValueError("Frozen train/validation manifest changed.")
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("testImagesIncluded") is not False:
    raise ValueError("Require a train/validation-only input.")
  classes = manifest["classes"]
  indices = {row["speciesId"]: i for i, row in enumerate(classes)}
  if (
    len(indices) != len(classes) or len(classes) != RECIPE["classCount"]
    or len({r["scientificName"] for r in classes}) != len(classes)
    or not set(indices).issubset(manifest["targetSpeciesIds"])
  ):
    raise ValueError("Invalid target class set.")
  splits = {"train": [], "validation": []}
  seen_ids, seen_sha, observations = set(), set(), set()
  authors = {"train": set(), "validation": set()}
  for row in manifest["samples"]:
    split = row["split"]
    if (
      split not in splits or row["speciesId"] not in indices
      or row["license"] not in {"cc0", "cc-by"} or not row.get("attribution")
      or not row.get("observerGroup") or row["sampleId"] in seen_ids
      or row["imageSha256"] in seen_sha or row["observationId"] in observations
      or row["scientificName"] != classes[indices[row["speciesId"]]]["scientificName"]
    ):
      raise ValueError("Test, duplicate, unlicensed or conflicting sample.")
    path = (data / row["path"]).resolve()
    if (
      not path.is_relative_to((data / "images").resolve())
      or sha256(path) != row["imageSha256"]
    ):
      raise ValueError("Raw image path or checksum changed.")
    seen_ids.add(row["sampleId"])
    seen_sha.add(row["imageSha256"])
    observations.add(row["observationId"])
    authors[split].add(row["observerGroup"])
    splits[split].append(row)
  if authors["train"] & authors["validation"]:
    raise ValueError("Author leakage between training and validation.")
  for split, expected in (("train", 70), ("validation", 15)):
    if Counter(r["speciesId"] for r in splits[split]) != dict.fromkeys(indices, expected):
      raise ValueError("Frozen per-species counts changed.")
  return classes, splits


def state_digest(state):
  digest = hashlib.sha256()
  for name, tensor in sorted(state.items()):
    digest.update(name.encode())
    digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def build_model(arm, *, count, weights=None, weights_sha=None, seed=None, head_seed=None):
  import torch
  from torchvision.models import resnet18

  if arm not in {"A", "B"}:
    raise ValueError("Unknown comparison arm.")
  with torch.random.fork_rng(), torch.device("cpu"):
    torch.manual_seed(RECIPE["seed"] if seed is None else seed)
    model = resnet18(weights=None, num_classes=count)
    if arm == "B":
      if weights is None or not weights_sha or sha256(weights) != weights_sha:
        raise ValueError("Require a locally pinned ImageNet checkpoint.")
      state = torch.load(weights, map_location="cpu", weights_only=True)
      if state["fc.weight"].shape != (1000, 512) or state["fc.bias"].shape != (1000,):
        raise ValueError("Unexpected source classification head.")
      # Never load or retain ImageNet's output layer.
      backbone = {k: v for k, v in state.items() if not k.startswith("fc.")}
      result = model.load_state_dict(backbone, strict=False)
      if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
        raise ValueError("Backbone checkpoint is incompatible.")
    torch.manual_seed(RECIPE["headSeed"] if head_seed is None else head_seed)
    model.fc = torch.nn.Linear(512, count)
  model.requires_grad_(True)
  return model


def transforms_for_comparison():
  from torchvision import transforms as T

  common = [
    T.ToTensor(), T.Normalize(RECIPE["mean"], RECIPE["std"]),
  ]
  train = T.Compose([
    T.RandomResizedCrop(
      224, scale=(0.7, 1.0), ratio=(0.75, 4 / 3),
      interpolation=T.InterpolationMode.BILINEAR, antialias=True,
    ),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    *common,
  ])
  validation = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.CenterCrop(224), *common,
  ])
  return train, validation


def summarize_predictions(predictions, classes):
  counts = {r["speciesId"]: [0, 0, 0] for r in classes}
  seen = set()
  for row in predictions:
    sid, ranking = row["speciesId"], row["top5"]
    if (
      row["sampleId"] in seen or not row.get("observerGroup") or sid not in counts
      or len(ranking) != min(5, len(classes)) or len(set(ranking)) != len(ranking)
      or not set(ranking).issubset(counts)
    ):
      raise ValueError("Invalid or duplicated validation prediction.")
    seen.add(row["sampleId"])
    counts[sid][0] += 1
    counts[sid][1] += int(ranking[0] == sid)
    counts[sid][2] += int(sid in ranking)
  if any(n == 0 for n, _, _ in counts.values()):
    raise ValueError("Validation must cover every class.")
  per_species = [{
    **row, "samples": counts[row["speciesId"]][0],
    "top1Recall": counts[row["speciesId"]][1] / counts[row["speciesId"]][0],
    "top5Recall": counts[row["speciesId"]][2] / counts[row["speciesId"]][0],
  } for row in classes]
  return {
    "samples": len(predictions), "perSpecies": per_species,
    "macroTop1": sum(r["top1Recall"] for r in per_species) / len(classes),
    "macroTop5": sum(r["top5Recall"] for r in per_species) / len(classes),
    "microTop1": sum(c[1] for c in counts.values()) / len(predictions),
    "microTop5": sum(c[2] for c in counts.values()) / len(predictions),
  }


def paired_comparison(a, b, classes, *, draws=2000, seed=2026092502):
  """Author-cluster paired bootstrap on validation; not independent test inference."""
  import numpy as np

  sa, sb = summarize_predictions(a, classes), summarize_predictions(b, classes)
  by_id = {r["sampleId"]: r for r in b}
  if {r["sampleId"] for r in a} != set(by_id):
    raise ValueError("Both arms must use exactly the same validation samples.")
  groups = {g: i for i, g in enumerate(sorted({r["observerGroup"] for r in a}))}
  indices = {r["speciesId"]: i for i, r in enumerate(classes)}
  totals = np.zeros((len(groups), len(classes)), dtype=np.float64)
  differences = np.zeros((2, len(groups), len(classes)), dtype=np.float64)
  transitions = {"lostTop5": 0, "gainedTop5": 0}
  for row in a:
    other = by_id[row["sampleId"]]
    if any(row[key] != other[key] for key in ("speciesId", "observerGroup", "imageSha256")):
      raise ValueError("Paired sample identity changed.")
    sid = row["speciesId"]
    g, c = groups[row["observerGroup"]], indices[sid]
    hit_a, hit_b = sid in row["top5"], sid in other["top5"]
    totals[g, c] += 1
    differences[0, g, c] += int(other["top5"][0] == sid) - int(row["top5"][0] == sid)
    differences[1, g, c] += int(hit_b) - int(hit_a)
    transitions["lostTop5"] += int(hit_a and not hit_b)
    transitions["gainedTop5"] += int(hit_b and not hit_a)
  rng = np.random.default_rng(seed)
  deltas, excluded = [], 0
  for _ in range(draws):
    weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
    denominator = weights @ totals
    if (denominator == 0).any():
      excluded += 1
      continue
    deltas.append([(weights @ d / denominator).mean() for d in differences])
  if not deltas:
    raise ValueError("No bootstrap draw covered all classes.")
  ci = np.quantile(np.array(deltas), [0.025, 0.975], axis=0)
  delta5 = sb["macroTop5"] - sa["macroTop5"]
  return {
    "A": sa, "B": sb,
    "deltaMacroTop1": sb["macroTop1"] - sa["macroTop1"], "deltaMacroTop5": delta5,
    "pairedTop5Transitions": transitions, "authorGroups": len(groups),
    "bootstrap": {
      "requestedDraws": draws, "usableDraws": len(deltas), "excludedMissingClass": excluded,
      "seed": seed, "unit": "observerGroup", "percentiles": [2.5, 97.5],
      "deltaTop1Interval": ci[:, 0].tolist(), "deltaTop5Interval": ci[:, 1].tolist(),
      "interpretation": (
        "Exploratory paired validation uncertainty, conditional on all classes present; "
        "not adjusted for validation-based checkpoint selection, not a final test CI."
      ),
    },
    "engineeringThreshold": 0.05, "worthConfirmation": delta5 >= 0.05,
    "automaticConfirmation": False, "independentFinalBenchmark": False,
    "productionApproved": False,
  }


def train_arm(data, experiment, arm):
  import numpy as np
  import torch
  import torchvision
  from PIL import Image
  from safetensors.torch import save_file
  from torch.utils.data import DataLoader, Dataset

  protocol_path = experiment / "protocol.json"
  protocol = json.loads(protocol_path.read_text())
  if (
    protocol["recipe"] != RECIPE or protocol["manifestSha256"] != MANIFEST_SHA
    or protocol["sourceSha256"] != sha256(Path(__file__))
    or protocol["torch"] != torch.__version__ or protocol["torchvision"] != torchvision.__version__
  ):
    raise ValueError("Experiment recipe, code or runtime changed.")
  output = experiment / arm
  if output.exists():
    raise ValueError("Refuse to overwrite or silently restart an arm.")
  started = time.monotonic()
  classes, splits = load_data(data)
  indices = {r["speciesId"]: i for i, r in enumerate(classes)}
  torch.set_num_threads(RECIPE["threads"])
  torch.set_num_interop_threads(1)
  torch.use_deterministic_algorithms(True)
  model = build_model(
    arm, count=len(classes), weights=Path(protocol["weightsPath"]),
    weights_sha=protocol["weightsSha256"],
  )
  initial_head = state_digest(model.fc.state_dict())
  initial_stem = model.conv1.weight.detach().clone()
  initial_backbone = state_digest({k: v for k, v in model.state_dict().items()
                                   if not k.startswith("fc.")})
  # Reset AFTER model construction. Initialization must not alter augmentation or batch RNG.
  seed = RECIPE["seed"]
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  augment, validation_transform = transforms_for_comparison()

  class Images(Dataset):
    def __init__(self, rows, transform):
      self.rows, self.transform = rows, transform

    def __len__(self):
      return len(self.rows)

    def __getitem__(self, index):
      row = self.rows[index]
      with Image.open(data / row["path"]) as image:
        tensor = self.transform(image.convert("RGB"))
      return tensor, indices[row["speciesId"]], index

  train_loader = DataLoader(
    Images(splits["train"], augment), batch_size=16, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(seed), drop_last=False,
  )
  val_loader = DataLoader(
    Images(splits["validation"], validation_transform), batch_size=16,
    shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed + 1),
  )
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
  reserve_space(experiment, 512 * 1024**2)
  output.mkdir()
  identity = {
    "arm": arm, "protocolSha256": sha256(protocol_path), "classes": classes,
    "initialHeadSha256": initial_head, "initialBackboneSha256": initial_backbone,
    "initialization": "random" if arm == "A" else "ResNet18_Weights.IMAGENET1K_V1",
    "pretrainedWeightsUsed": arm == "B", "pretrainedFeaturesUsed": False,
    "inheritedClassCount": 0, "trainBackbone": True,
    "externalSpeciesFallbackAllowed": False, "productionApproved": False,
  }
  write_json(output / "plan.json", identity)
  history, best, stale, steps = [], math.inf, 0, 0
  best_epoch = 0
  for epoch in range(1, 31):
    model.train()
    train_loss, train_total = 0.0, 0
    consumed = []
    order_hash, augmentation_hash = hashlib.sha256(), hashlib.sha256()
    for batch, (images, labels, positions) in enumerate(train_loader, 1):
      # Exact input traces detect differences in all batch ordering AND augmented pixels.
      order_hash.update(positions.numpy().tobytes())
      augmentation_hash.update(images.numpy().tobytes())
      consumed.extend(positions.tolist())
      optimizer.zero_grad(set_to_none=True)
      loss = torch.nn.functional.cross_entropy(model(images), labels)
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss.")
      loss.backward()
      if model.conv1.weight.grad is None:
        raise ValueError("Backbone was not trained.")
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
      optimizer.step()
      steps += 1
      train_total += len(labels)
      train_loss += float(loss.detach()) * len(labels)
      if batch == 1 or batch % 10 == 0:
        progress = {
          "updatedAt": now(), "arm": arm, "stage": "training", "epoch": epoch,
          "maxEpochs": 30, "batch": batch, "batches": len(train_loader),
          "trainLoss": train_loss / train_total, "elapsedSeconds": time.monotonic() - started,
          "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if sorted(consumed) != list(range(len(splits["train"]))):
      raise ValueError("Not every training sample was consumed exactly once.")
    model.eval()
    predictions, val_loss = [], 0.0
    with torch.inference_mode():
      for images, labels, positions in val_loader:
        logits = model(images)
        val_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
        rankings = logits.topk(5, dim=1).indices.tolist()
        for position, ranking in zip(positions.tolist(), rankings, strict=True):
          row = splits["validation"][position]
          predictions.append({
            **{k: row[k] for k in ("sampleId", "imageSha256", "speciesId", "observerGroup")},
            "top5": [classes[i]["speciesId"] for i in ranking],
          })
    score = val_loss / len(predictions)
    if not math.isfinite(score):
      raise ValueError("Nonfinite validation loss.")
    metrics = summarize_predictions(predictions, classes)
    summary = {
      "epoch": epoch, "trainLoss": train_loss / train_total, "validationLoss": score,
      **{k: v for k, v in metrics.items() if k != "perSpecies"},
      "learningRate": optimizer.param_groups[0]["lr"],
      "elapsedSeconds": time.monotonic() - started,
      "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      "batchOrderSha256": order_hash.hexdigest(),
      "augmentedPixelsSha256": augmentation_hash.hexdigest(),
    }
    history.append(summary)
    if score < best:
      best, best_epoch, stale = score, epoch, 0
      reserve_space(output, 128 * 1024**2)
      temporary = output / "best.safetensors.tmp"
      save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, temporary)
      os.replace(temporary, output / "classifier.safetensors")
      write_json(output / "validation-predictions.json", predictions)
      write_json(output / "validation.json", {**metrics, "epoch": epoch, "loss": score})
    else:
      stale += 1
    scheduler.step()
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      **summary, "updatedAt": now(), "arm": arm, "stage": "epoch-complete",
      "bestEpoch": best_epoch, "staleEpochs": stale,
    })
    print(json.dumps({"arm": arm, **summary}), flush=True)
    if stale >= 10:
      break
  if torch.equal(model.conv1.weight.detach(), initial_stem):
    raise ValueError("Backbone did not change.")
  completed = {
    **identity, "status": "trained-offline", "completedAt": now(),
    "checkpointSha256": sha256(output / "classifier.safetensors"),
    "epochsRun": len(history), "bestEpoch": best_epoch, "optimizerSteps": steps,
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "validation": json.loads((output / "validation.json").read_text()),
    "independentFinalBenchmark": False, "confidenceCalibrated": False,
    "scope": "Offline initialization experiment, not a serving solution or binary classifier.",
  }
  # Support metadata exists only after full training; no serving registry is updated.
  write_json(output / "trained-species.json", {
    "schemaVersion": "initialization-experiment-v1",
    **{k: completed[k] for k in (
      "status", "initialization", "checkpointSha256", "protocolSha256",
      "pretrainedWeightsUsed", "inheritedClassCount", "productionApproved",
    )},
    "classCount": len(classes),
    "classes": [{**r, "trainingExamples": 70, "classRole": "trained-target"} for r in classes],
  })
  write_json(output / "model.json", completed)
  write_json(output / "progress.json", {
    "stage": "trained-offline", "arm": arm, "updatedAt": now(), "bestEpoch": best_epoch,
    "epochsRun": len(history), "onlineChanged": False,
  })


def compare_completed(experiment):
  protocol_sha = sha256(experiment / "protocol.json")
  models, histories, predictions = {}, {}, {}
  for arm in ("A", "B"):
    root = experiment / arm
    models[arm] = json.loads((root / "model.json").read_text())
    if (
      models[arm]["status"] != "trained-offline"
      or models[arm]["protocolSha256"] != protocol_sha
      or models[arm]["checkpointSha256"] != sha256(root / "classifier.safetensors")
    ):
      raise ValueError("Incomplete, changed or incompatible experiment.")
    histories[arm] = json.loads((root / "history.json").read_text())
    predictions[arm] = json.loads((root / "validation-predictions.json").read_text())
  if (
    models["A"]["classes"] != models["B"]["classes"]
    or models["A"]["initialHeadSha256"] != models["B"]["initialHeadSha256"]
    or models["A"]["initialBackboneSha256"] == models["B"]["initialBackboneSha256"]
  ):
    raise ValueError("Initial models violate the A/B design.")
  shared_epochs = min(len(histories["A"]), len(histories["B"]))
  for a, b in zip(histories["A"], histories["B"]):
    if any(a[k] != b[k] for k in ("epoch", "batchOrderSha256", "augmentedPixelsSha256")):
      raise ValueError("Batch sequence or augmentation differed between arms.")
  report = paired_comparison(predictions["A"], predictions["B"], models["A"]["classes"])
  report.update({
    "protocolSha256": protocol_sha, "sharedEpochsWithIdenticalInputs": shared_epochs,
    "resources": {a: {k: models[a][k] for k in (
      "bestEpoch", "epochsRun", "elapsedSeconds", "peakRssBytes",
    )} for a in ("A", "B")},
  })
  write_json(experiment / "comparison.json", report)
  return report


def run_pair(data, experiment, weights):
  import torch
  import torchvision
  from torchvision.models import ResNet18_Weights

  if experiment.exists():
    raise ValueError("Use a new experiment directory; no implicit restart.")
  weights_sha = sha256(weights)
  if (
    not weights_sha.startswith("f37072fd")
    or ResNet18_Weights.IMAGENET1K_V1.url != WEIGHTS_URL
  ):
    raise ValueError("Require the fixed official ImageNet V1 artifact.")
  classes, splits = load_data(data)
  reserve_space(experiment.parent, 512 * 1024**2)
  experiment.mkdir()
  write_json(experiment / "protocol.json", {
    "createdAt": now(), "recipe": RECIPE,
    "authorization": "User approved this fixed ImageNet-vs-random 82-class comparison.",
    "sourceDatasetPolicy": "Original scratch snapshot is unchanged; this protocol authorizes reuse.",
    "dataPath": str(data.resolve()), "manifestSha256": MANIFEST_SHA, "classes": classes,
    "counts": {k: len(v) for k, v in splits.items()},
    "weightsPath": str(weights.resolve()), "weightsUrl": WEIGHTS_URL,
    "weightsSha256": weights_sha, "sourceSha256": sha256(Path(__file__)),
    "torch": torch.__version__, "torchvision": torchvision.__version__,
    "execution": "A then B, separate sequential processes, no automatic extra experiments",
  })
  try:
    for arm in ("A", "B"):
      write_json(experiment / "status.json", {
        "stage": "running", "activeArm": arm, "updatedAt": now(), "onlineChanged": False,
      })
      subprocess.run([
        sys.executable, "-m", "yizhu_server.initialization_comparison",
        "--data", str(data), "--experiment", str(experiment), "--arm", arm,
      ], check=True)
    compare_completed(experiment)
    write_json(experiment / "status.json", {
      "stage": "comparison-complete", "updatedAt": now(), "onlineChanged": False,
    })
  except BaseException as exc:
    write_json(experiment / "status.json", {
      "stage": "failed", "errorType": type(exc).__name__, "updatedAt": now(),
      "onlineChanged": False,
    })
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--experiment", type=Path, required=True)
  parser.add_argument("--weights", type=Path)
  parser.add_argument("--arm", choices=["A", "B"])
  args = parser.parse_args()
  if args.arm:
    train_arm(args.data.resolve(), args.experiment.resolve(), args.arm)
  elif args.weights:
    run_pair(args.data.resolve(), args.experiment.resolve(), args.weights.resolve())
  else:
    parser.error("--weights is required when starting a pair")


if __name__ == "__main__":
  main()
