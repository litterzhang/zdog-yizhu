"""Stage B: fine-tune layer4 and the trained target head, never deploy automatically."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import resource
import time
from collections import Counter
from pathlib import Path

from yizhu_server.initialization_comparison import (
  MANIFEST_SHA,
  load_data,
  paired_comparison,
  state_digest,
  summarize_predictions,
  transforms_for_comparison,
)
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import now, sha256, write_json

PARENT_SHA = "7bb8ecdfc0731fd8e682d755f2d33caaf52c1333608ec28c7771c1771b067f91"
PARENT_PROTOCOL_SHA = "6dc638286e9ef714463349ccdbbf0f5ab97cd5eca616e53a3df8bd7d45bb77a2"
RECIPE = {
  "architecture": "resnet18", "classCount": 82, "seed": 20260926,
  "epochs": 15, "patience": 5, "batchSize": 16, "threads": 2,
  "optimizer": "AdamW", "layer4LearningRate": 0.0001, "headLearningRate": 0.001,
  "weightDecay": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8,
  "scheduler": None, "gradientClipNorm": 5.0,
  "trainable": ["layer4 convolutions", "fc"], "batchNormAffineFrozen": True,
  "batchNormStatisticsFrozen": True, "earlierLayersFrozen": True,
  "imageSize": 224, "validationResizeShortEdge": 256,
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
  "interpolation": "bilinear", "antialias": True,
  "trainAugmentation": {
    "randomResizedCropScale": [0.7, 1.0], "randomResizedCropRatio": [0.75, 4 / 3],
    "horizontalFlip": 0.5,
    "colorJitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.02},
  },
  "validationCrop": "center224", "loss": "cross entropy",
  "selection": "lowest validation CE among trained epochs; earliest exact tie",
  "baseline": "stage A best epoch 8, reference only; report regressions honestly",
  "pairedBootstrap": {"unit": "observerGroup", "draws": 2000, "seed": 2026092602},
  "engineeringThresholdTop5": 0.05, "automaticConfirmation": False,
  "testUsed": False, "automaticStageC": False, "automaticDeployment": False,
}


def run_recipe(confirmation=False):
  """One prespecified seed check, not a configurable hyperparameter search."""
  if type(confirmation) is not bool:
    raise ValueError("Confirmation must be an explicit boolean.")
  recipe = copy.deepcopy(RECIPE)
  if confirmation:
    recipe["seed"] = 20260927
  return recipe


def configure_partial_training(model):
  """Freeze all BN affine parameters too; only layer4 convolution weights and fc train."""
  import torch

  model.requires_grad_(False)
  model.layer4.requires_grad_(True)
  model.fc.requires_grad_(True)
  for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.requires_grad_(False)
  set_partial_mode(model)
  return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def set_partial_mode(model):
  import torch

  model.eval()
  model.layer4.train()
  model.fc.train()
  for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.eval()


def frozen_digest(model):
  trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
  return state_digest({k: v for k, v in model.state_dict().items() if k not in trainable})


def validate_parent(parent, classes):
  checkpoint = parent / "classifier.safetensors"
  meta = json.loads((parent / "model.json").read_text())
  protocol = json.loads((parent / "protocol.json").read_text())
  support = json.loads((parent / "trained-species.json").read_text())
  if (
    sha256(checkpoint) != PARENT_SHA or meta["weightsSha256"] != PARENT_SHA
    or support["weightsSha256"] != PARENT_SHA
    or sha256(parent / "protocol.json") != PARENT_PROTOCOL_SHA
    or meta["protocolSha256"] != PARENT_PROTOCOL_SHA
    or protocol["manifestSha256"] != MANIFEST_SHA
    or meta["status"] != "trained-offline" or meta["testUsed"] is not False
    or meta["architecture"] != "resnet18" or meta["bestEpoch"] != 8
    or meta["productionApproved"] is not False
    or support["classes"] != meta["classes"]
    or protocol["classes"] != classes
    or len(meta["classes"]) != len(classes)
  ):
    raise ValueError("Parent is not the pinned, completed stage A checkpoint.")
  for row, expected in zip(meta["classes"], classes, strict=True):
    if (
      any(row[k] != expected[k] for k in ("speciesId", "scientificName", "chineseName"))
      or row["trainingExamples"] != 70
    ):
      raise ValueError("Parent class mapping or training coverage differs.")
  return meta


def make_loaders(data, splits, classes, recipe):
  import torch
  from PIL import Image
  from torch.utils.data import DataLoader, Dataset

  indices = {r["speciesId"]: i for i, r in enumerate(classes)}
  train_transform, validation_transform = transforms_for_comparison()

  class Images(Dataset):
    def __init__(self, split, transform):
      self.rows, self.transform = splits[split], transform
      if any(r["split"] != split for r in self.rows):
        raise ValueError("Test or misplaced samples cannot enter training/evaluation.")

    def __len__(self):
      return len(self.rows)

    def __getitem__(self, index):
      row = self.rows[index]
      with Image.open(data / row["path"]) as image:
        tensor = self.transform(image.convert("RGB"))
      return tensor, indices[row["speciesId"]], index

  kwargs = {"batch_size": recipe["batchSize"], "num_workers": 0, "drop_last": False}
  return {
    "train": DataLoader(
      Images("train", train_transform), shuffle=True,
      generator=torch.Generator().manual_seed(recipe["seed"]), **kwargs,
    ),
    "validation": DataLoader(
      Images("validation", validation_transform), shuffle=False,
      generator=torch.Generator().manual_seed(recipe["seed"] + 1), **kwargs,
    ),
    "trainEvaluation": DataLoader(
      Images("train", validation_transform), shuffle=False,
      generator=torch.Generator().manual_seed(recipe["seed"] + 2), **kwargs,
    ),
  }


def evaluate(model, loader, rows, classes):
  import torch

  model.eval()
  predictions, positions_seen, loss_sum = [], [], 0.0
  with torch.inference_mode():
    for images, labels, positions in loader:
      logits = model(images)
      if logits.shape != (len(labels), len(classes)) or not torch.isfinite(logits).all():
        raise ValueError("Invalid model output.")
      loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
      rankings = logits.topk(min(5, len(classes)), dim=1).indices.tolist()
      for pos, label, ranking in zip(positions.tolist(), labels.tolist(), rankings, strict=True):
        row = rows[pos]
        if row["speciesId"] != classes[label]["speciesId"]:
          raise ValueError("Evaluation labels and image index differ.")
        positions_seen.append(pos)
        predictions.append({
          **{k: row[k] for k in ("sampleId", "speciesId", "imageSha256", "observerGroup")},
          "top5": [classes[i]["speciesId"] for i in ranking],
        })
  if positions_seen != list(range(len(rows))):
    raise ValueError("Evaluation must consume every sample exactly once in order.")
  metrics = summarize_predictions(predictions, classes)
  metrics["loss"] = loss_sum / len(rows)
  if not math.isfinite(metrics["loss"]):
    raise ValueError("Nonfinite validation loss.")
  return metrics, predictions


def fit(model, data, output, splits, classes, *, recipe=None, expected_baseline=None):
  """A bounded single run, callable with a small synthetic model for tests."""
  import torch
  from safetensors.torch import load_file, save_file

  recipe = RECIPE if recipe is None else recipe
  names = configure_partial_training(model)
  if (
    not names or not any(n.startswith("layer4.") for n in names)
    or not any(n.startswith("fc.") for n in names)
    or any(not n.startswith(("layer4.", "fc.")) for n in names)
  ):
    raise ValueError("Unexpected trainable parameter set.")
  before = frozen_digest(model)
  initial_layer4 = state_digest(model.layer4.state_dict())
  initial_head = state_digest(model.fc.state_dict())
  write_json(output / "trainable-parameters.json", {
    "names": names, "count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "frozenSha256": before,
  })
  loaders = make_loaders(data, splits, classes, recipe)
  optimizer = torch.optim.AdamW([
    {"params": [p for p in model.layer4.parameters() if p.requires_grad],
     "lr": recipe["layer4LearningRate"]},
    {"params": list(model.fc.parameters()), "lr": recipe["headLearningRate"]},
  ], weight_decay=recipe["weightDecay"], betas=tuple(recipe["betas"]), eps=recipe["eps"])
  history, best_loss, best_epoch, stale, steps = [], math.inf, 0, 0, 0
  started = time.monotonic()
  # One deterministic raw-image baseline verifies that stage A is the actual starting model.
  baseline, baseline_predictions = evaluate(
    model, loaders["validation"], splits["validation"], classes,
  )
  if expected_baseline is not None:
    for key in ("macroTop1", "macroTop5", "loss"):
      if abs(baseline[key] - expected_baseline[key]) > 1e-5:
        raise ValueError("Raw-image baseline does not reproduce stage A validation.")
  write_json(output / "baseline-validation.json", baseline)
  write_json(output / "baseline-predictions.json", baseline_predictions)
  for epoch in range(1, recipe["epochs"] + 1):
    set_partial_mode(model)
    consumed, train_loss, seen = [], 0.0, 0
    order_hash = hashlib.sha256()
    for batch, (images, labels, positions) in enumerate(loaders["train"], 1):
      order_hash.update(positions.numpy().tobytes())
      consumed.extend(positions.tolist())
      optimizer.zero_grad(set_to_none=True)
      loss = torch.nn.functional.cross_entropy(model(images), labels)
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss.")
      loss.backward()
      for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
          raise ValueError(f"Missing gradient for trainable parameter: {name}")
        if not parameter.requires_grad and parameter.grad is not None:
          raise ValueError("Frozen parameter received gradient.")
      torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        recipe["gradientClipNorm"], error_if_nonfinite=True,
      )
      optimizer.step()
      steps += 1
      seen += len(labels)
      train_loss += float(loss.detach()) * len(labels)
      if batch == 1 or batch % 10 == 0:
        progress = {
          "stage": "training", "updatedAt": now(), "epoch": epoch,
          "maxEpochs": recipe["epochs"], "batch": batch, "batches": len(loaders["train"]),
          "trainLoss": train_loss / seen, "elapsedSeconds": time.monotonic() - started,
          "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if sorted(consumed) != list(range(len(splits["train"]))):
      raise ValueError("Training sample coverage differs.")
    if frozen_digest(model) != before:
      raise ValueError("Frozen layers or BatchNorm parameters/buffers changed.")
    metrics, predictions = evaluate(model, loaders["validation"], splits["validation"], classes)
    summary = {
      "epoch": epoch, "trainOnlineLoss": train_loss / seen, "validationLoss": metrics["loss"],
      **{k: metrics[k] for k in ("macroTop1", "macroTop5", "microTop1", "microTop5")},
      "batchOrderSha256": order_hash.hexdigest(),
      "elapsedSeconds": time.monotonic() - started,
      "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    history.append(summary)
    if metrics["loss"] < best_loss:
      best_loss, best_epoch, stale = metrics["loss"], epoch, 0
      reserve_space(output, 64 * 1024**2)
      temporary = output / "best.safetensors.tmp"
      save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, temporary)
      os.replace(temporary, output / "classifier.safetensors")
      write_json(output / "validation.json", {**metrics, "epoch": epoch})
      write_json(output / "validation-predictions.json", predictions)
    else:
      stale += 1
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      **summary, "stage": "epoch-complete", "updatedAt": now(),
      "bestEpoch": best_epoch, "staleEpochs": stale,
    })
    print(json.dumps(summary), flush=True)
    if stale >= recipe["patience"]:
      break
  model.load_state_dict(load_file(output / "classifier.safetensors"), strict=True)
  if frozen_digest(model) != before:
    raise ValueError("Selected checkpoint violates frozen parameters.")
  if (
    state_digest(model.layer4.state_dict()) == initial_layer4
    or state_digest(model.fc.state_dict()) == initial_head
  ):
    raise ValueError("Selected checkpoint did not train both layer4 and the head.")
  train_metrics, _ = evaluate(model, loaders["trainEvaluation"], splits["train"], classes)
  write_json(output / "train-selected.json", train_metrics)
  comparison = paired_comparison(
    baseline_predictions,
    json.loads((output / "validation-predictions.json").read_text()), classes,
    draws=recipe["pairedBootstrap"]["draws"], seed=recipe["pairedBootstrap"]["seed"],
  )
  comparison["scope"] = (
    "Stage A baseline versus stage B selected checkpoint on validation, not an "
    "initialization-only comparison; augmentation and trainable layers also changed."
  )
  write_json(output / "stage-a-comparison.json", comparison)
  return {
    "bestEpoch": best_epoch, "epochsRun": len(history), "optimizerSteps": steps,
    "frozenParametersAndBatchNormUnchanged": True, "frozenSha256": before,
    "initialLayer4Sha256": initial_layer4, "initialHeadSha256": initial_head,
    "baselineValidation": {k: v for k, v in baseline.items() if k != "perSpecies"},
  }


def train(data, parent, output, *, confirmation=False):
  import numpy as np
  import torch
  import torchvision
  from safetensors.torch import load_file
  from torchvision.models import resnet18

  if output.exists():
    raise ValueError("Use a new output directory; no overwrite or implicit restart.")
  recipe = run_recipe(confirmation)
  started = time.monotonic()
  classes, splits = load_data(data)
  parent_meta = validate_parent(parent, classes)
  reserve_space(output.parent, 256 * 1024**2)
  output.mkdir()
  write_json(output / "protocol.json", {
    "createdAt": now(), "stage": "B-layer4-finetune", "recipe": recipe,
    "confirmationRun": confirmation,
    "authorization": (
      "User said continue after the proposed single fixed-recipe seed confirmation."
      if confirmation else
      "User explicitly said go after reviewing stage A and stage B proposal."
    ),
    "manifestSha256": MANIFEST_SHA, "parentCheckpointSha256": PARENT_SHA,
    "parentProtocolSha256": PARENT_PROTOCOL_SHA, "parentBestEpoch": 8,
    "classes": classes, "counts": {k: len(v) for k, v in splits.items()},
    "torch": torch.__version__, "torchvision": torchvision.__version__,
    "sourceSha256": {
      name: sha256(Path(__file__).with_name(name)) for name in (
        "partial_species_training.py", "initialization_comparison.py",
        "snapshot_files.py", "training_data.py",
      )
    },
    "licenseScope": "Same offline research-only scope as stage A; deployment rights not cleared.",
    "testUsed": False, "automaticDeployment": False,
  })
  write_json(output / "progress.json", {"stage": "initializing", "updatedAt": now()})
  try:
    torch.set_num_threads(recipe["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    model = resnet18(weights=None, num_classes=len(classes))
    model.load_state_dict(load_file(parent / "classifier.safetensors"), strict=True)
    torch.manual_seed(recipe["seed"])
    fitted = fit(
      model, data, output, splits, classes, recipe=recipe,
      expected_baseline=parent_meta["validation"],
    )
    counts = Counter(r["speciesId"] for r in splits["train"])
    metadata = {
      **fitted, "status": "trained-offline", "completedAt": now(),
      "architecture": "resnet18", "classCount": len(classes),
      "weightsSha256": sha256(output / "classifier.safetensors"),
      "parentWeightsSha256": PARENT_SHA,
      "protocolSha256": sha256(output / "protocol.json"),
      "preprocessing": {
        k: v for k, v in parent_meta["preprocessing"].items()
        if k not in {"fixedViewsPerImage", "randomImageAugmentation"}
      },
      "trainingAugmentation": recipe["trainAugmentation"],
      "trainingSeed": recipe["seed"], "confirmationRun": confirmation,
      "pretrainedWeightsUsed": True, "pretrainedFeaturesUsed": False,
      "classes": [{**r, "trainingExamples": counts[r["speciesId"]],
                   "classRole": "trained-partial-target"} for r in classes],
      "validation": json.loads((output / "validation.json").read_text()),
      "elapsedSeconds": time.monotonic() - started,
      "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      "inheritedClassCount": 0, "externalSpeciesFallbackAllowed": False,
      "confidenceCalibrated": False, "testUsed": False,
      "independentFinalBenchmark": False, "productionApproved": False,
      "includesBinaryClassifier": False, "automaticStageC": False,
    }
    write_json(output / "trained-species.json", {
      "schemaVersion": "partial-head-experiment-v1",
      **{k: metadata[k] for k in (
        "status", "classCount", "classes", "weightsSha256", "protocolSha256",
        "parentWeightsSha256", "pretrainedWeightsUsed", "inheritedClassCount",
        "productionApproved",
      )},
    })
    write_json(output / "model.json", metadata)
    write_json(output / "progress.json", {
      "stage": "trained-offline", "updatedAt": now(), **fitted, "onlineChanged": False,
    })
  except BaseException as exc:
    write_json(output / "failure.json", {
      "stage": "failed", "updatedAt": now(), "errorType": type(exc).__name__,
    })
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--parent", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--confirmation", action="store_true",
                      help="Run the single prespecified training seed 20260927.")
  args = parser.parse_args()
  train(args.data.resolve(), args.parent.resolve(), args.output.resolve(),
        confirmation=args.confirmation)


if __name__ == "__main__":
  main()
