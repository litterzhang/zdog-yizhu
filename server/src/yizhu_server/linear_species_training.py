"""Stage A: fresh ImageNet features and a new species head; offline research only."""

import argparse
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
  WEIGHTS_URL,
  build_model,
  load_data,
  state_digest,
  summarize_predictions,
)
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import now, sha256, write_json

WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RECIPE = {
  "architecture": "resnet18", "weights": "ResNet18_Weights.IMAGENET1K_V1",
  "classCount": 82, "featureDimensions": 512, "headParameters": 42066,
  "seed": 20260925, "headSeed": 2026092501, "threads": 2,
  "batchSize": 16, "featureBatchSize": 16, "epochs": 50, "patience": 8,
  "optimizer": "AdamW", "learningRate": 0.001, "weightDecay": 0.0001,
  "betas": [0.9, 0.999], "eps": 1e-8, "scheduler": None, "gradientClipping": None,
  "loss": "cross entropy", "selection": "lowest validation CE; earliest exact tie",
  "backboneFrozen": True, "backboneMode": "eval", "batchNormStatisticsFrozen": True,
  "preprocessing": {
    "resizeShortEdge": 256, "centerCrop": 224, "interpolation": "bilinear",
    "antialias": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "fixedViewsPerImage": 1, "randomImageAugmentation": False,
  },
  "testUsed": False, "automaticStageB": False, "automaticDeployment": False,
}


def split_frozen_backbone(model):
  import torch

  head = model.fc
  model.fc = torch.nn.Identity()
  model.requires_grad_(False).eval()
  head.requires_grad_(True).train()
  return model, head


def extract_features(backbone, data, rows, indices, transform, *, batch_size=16, progress=None):
  import torch
  from PIL import Image
  from torch.utils.data import DataLoader, Dataset

  if any(p.requires_grad for p in backbone.parameters()) or any(
    module.training for module in backbone.modules()
  ):
    raise ValueError("Feature extraction requires a fully frozen evaluation-mode backbone.")

  class Images(Dataset):
    def __len__(self):
      return len(rows)

    def __getitem__(self, index):
      row = rows[index]
      if row["split"] not in {"train", "validation"}:
        raise ValueError("Test images cannot enter the feature cache.")
      with Image.open(data / row["path"]) as image:
        return transform(image.convert("RGB")), indices[row["speciesId"]]

  # Allocate outside inference_mode so cached features remain legal autograd inputs to the head.
  features = torch.empty((len(rows), 512), dtype=torch.float32)
  labels = torch.empty(len(rows), dtype=torch.int64)
  loader = DataLoader(Images(), batch_size=batch_size, shuffle=False, num_workers=0)
  offset = 0
  with torch.inference_mode():
    for images, target in loader:
      encoded = backbone(images)
      if encoded.shape != (len(target), 512) or not torch.isfinite(encoded).all():
        raise ValueError("Invalid frozen features.")
      features[offset:offset + len(target)].copy_(encoded)
      labels[offset:offset + len(target)].copy_(target)
      offset += len(target)
      if progress:
        progress(offset, len(rows))
  if offset != len(rows):
    raise ValueError("Incomplete feature extraction.")
  return features, labels


def evaluate_head(head, features, labels, rows, classes):
  import torch

  if len(rows) != len(features) or len(rows) != len(labels):
    raise ValueError("Feature rows and labels differ.")
  indices = {r["speciesId"]: i for i, r in enumerate(classes)}
  if labels.tolist() != [indices[r["speciesId"]] for r in rows]:
    raise ValueError("Feature labels do not match the sample index.")
  head.eval()
  loss, rankings = 0.0, []
  with torch.inference_mode():
    for offset in range(0, len(labels), 256):
      logits = head(features[offset:offset + 256])
      loss += float(torch.nn.functional.cross_entropy(
        logits, labels[offset:offset + 256], reduction="sum",
      ))
      rankings.extend(logits.topk(min(5, len(classes)), dim=1).indices.tolist())
  predictions = [{
    **{k: row[k] for k in ("sampleId", "speciesId", "imageSha256", "observerGroup")},
    "top5": [classes[i]["speciesId"] for i in ranking],
  } for row, ranking in zip(rows, rankings, strict=True)]
  metrics = summarize_predictions(predictions, classes)
  metrics["loss"] = loss / len(labels)
  if not math.isfinite(metrics["loss"]):
    raise ValueError("Nonfinite head evaluation loss.")
  return metrics, predictions


def fit_head(head, tensors, splits, classes, output, *, recipe=None):
  import torch
  from safetensors.torch import load_file, save_file
  from torch.utils.data import DataLoader, TensorDataset

  recipe = RECIPE if recipe is None else recipe
  for split in ("train", "validation"):
    if any(r["split"] != split for r in splits[split]):
      raise ValueError("Test or misplaced rows in head training.")
    # Validate complete sample alignment before the first optimizer update.
    evaluate_head(head, tensors[f"{split}Features"], tensors[f"{split}Labels"],
                  splits[split], classes)
  initial_head = state_digest(head.state_dict())
  generator = torch.Generator().manual_seed(recipe["seed"])
  loader = DataLoader(
    TensorDataset(tensors["trainFeatures"], tensors["trainLabels"]),
    batch_size=recipe["batchSize"], shuffle=True, generator=generator, num_workers=0,
  )
  optimizer = torch.optim.AdamW(
    head.parameters(), lr=recipe["learningRate"], weight_decay=recipe["weightDecay"],
    betas=tuple(recipe["betas"]), eps=recipe["eps"],
  )
  started = time.monotonic()
  history, best_loss, best_epoch, stale, steps = [], math.inf, 0, 0, 0
  for epoch in range(1, recipe["epochs"] + 1):
    head.train()
    loss_sum, examples = 0.0, 0
    for features, labels in loader:
      optimizer.zero_grad(set_to_none=True)
      loss = torch.nn.functional.cross_entropy(head(features), labels)
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite head training loss.")
      loss.backward()
      optimizer.step()
      steps += 1
      loss_sum += float(loss.detach()) * len(labels)
      examples += len(labels)
    if examples != len(splits["train"]):
      raise ValueError("Incomplete head epoch.")
    metrics, predictions = evaluate_head(
      head, tensors["validationFeatures"], tensors["validationLabels"],
      splits["validation"], classes,
    )
    summary = {
      "epoch": epoch, "trainOnlineLoss": loss_sum / examples,
      "validationLoss": metrics["loss"],
      **{k: metrics[k] for k in ("macroTop1", "macroTop5", "microTop1", "microTop5")},
      "headTrainingSeconds": time.monotonic() - started,
    }
    history.append(summary)
    if metrics["loss"] < best_loss:
      best_loss, best_epoch, stale = metrics["loss"], epoch, 0
      temporary = output / "head.safetensors.tmp"
      save_file({k: v.detach().contiguous() for k, v in head.state_dict().items()}, temporary)
      os.replace(temporary, output / "head.safetensors")
      write_json(output / "validation.json", {**metrics, "epoch": epoch})
      write_json(output / "validation-predictions.json", predictions)
    else:
      stale += 1
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      **summary, "stage": "head-training", "updatedAt": now(),
      "bestEpoch": best_epoch, "staleEpochs": stale,
    })
    print(json.dumps(summary), flush=True)
    if stale >= recipe["patience"]:
      break
  head.load_state_dict(load_file(output / "head.safetensors"))
  if initial_head == state_digest(head.state_dict()):
    raise ValueError("Selected head did not change from initialization.")
  train_metrics, _ = evaluate_head(
    head, tensors["trainFeatures"], tensors["trainLabels"], splits["train"], classes,
  )
  write_json(output / "train-selected.json", train_metrics)
  return {
    "bestEpoch": best_epoch, "epochsRun": len(history), "optimizerSteps": steps,
    "initialHeadSha256": initial_head, "headTrainingSeconds": time.monotonic() - started,
  }


def train(data, output, weights):
  import numpy as np
  import torch
  import torchvision
  from safetensors.torch import load_file, save_file
  from torchvision.models import ResNet18_Weights

  if output.exists():
    raise ValueError("Use a new output directory; no overwrite or implicit resume.")
  if sha256(weights) != WEIGHTS_SHA or ResNet18_Weights.IMAGENET1K_V1.url != WEIGHTS_URL:
    raise ValueError("Require the pinned official ImageNet ResNet18 V1 weights.")
  started = time.monotonic()
  classes, splits = load_data(data)
  reserve_space(output.parent, 256 * 1024**2)
  output.mkdir()
  protocol = {
    "createdAt": now(), "stage": "A-frozen-linear-head", "recipe": RECIPE,
    "authorization": "User approved staged transfer learning; stage B is not automatic.",
    "manifestSha256": MANIFEST_SHA, "classes": classes,
    "counts": {k: len(v) for k, v in splits.items()},
    "weightsUrl": WEIGHTS_URL, "weightsSha256": WEIGHTS_SHA,
    "torch": torch.__version__, "torchvision": torchvision.__version__,
    "sourceSha256": {
      name: sha256(Path(__file__).with_name(name)) for name in (
        "linear_species_training.py", "initialization_comparison.py",
        "training_data.py", "snapshot_files.py",
      )
    },
    "licenseReview": {
      "scope": "Offline research only; not approval for product/commercial deployment.",
      "torchvisionCode": "BSD-3-Clause does not alone establish pretrained weight rights.",
      "weightsNotice": "Torchvision notes weights may carry dataset-derived terms.",
      "imagenetDatabase": "Official database terms limit use to non-commercial research/education.",
      "deploymentRightsCleared": False,
      "references": [
        "https://github.com/pytorch/vision/blob/main/LICENSE",
        "https://github.com/pytorch/vision#pre-trained-model-license",
        "https://www.image-net.org/download.php",
      ],
    },
    "independentFinalBenchmark": False, "productionApproved": False,
  }
  write_json(output / "protocol.json", protocol)
  try:
    torch.set_num_threads(RECIPE["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RECIPE["seed"])
    random.seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    model = build_model(
      "B", count=len(classes), weights=weights, weights_sha=WEIGHTS_SHA,
      seed=RECIPE["seed"], head_seed=RECIPE["headSeed"],
    )
    backbone, head = split_frozen_backbone(model)
    if sum(p.numel() for p in head.parameters()) != RECIPE["headParameters"]:
      raise ValueError("Unexpected head parameter count.")
    backbone_before = state_digest(backbone.state_dict())
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    if (
      transform.crop_size != [224] or transform.resize_size != [256]
      or transform.mean != RECIPE["preprocessing"]["mean"]
      or transform.std != RECIPE["preprocessing"]["std"]
      or transform.interpolation.value != "bilinear" or transform.antialias is not True
    ):
      raise ValueError("Official preprocessing differs from the frozen protocol.")
    indices = {r["speciesId"]: i for i, r in enumerate(classes)}
    tensors, feature_index = {}, {}
    feature_start = time.monotonic()
    for split in ("train", "validation"):
      def progress(done, total):
        if done == total or done % 160 == 0 or done == 16:
          status = {
            "stage": "extracting-features", "split": split,
            "completedImages": done, "totalImages": total,
            "updatedAt": now(), "elapsedSeconds": time.monotonic() - started,
            "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          }
          write_json(output / "progress.json", status)
          print(json.dumps(status), flush=True)
      features, labels = extract_features(
        backbone, data, splits[split], indices, transform, progress=progress,
        batch_size=RECIPE["featureBatchSize"],
      )
      tensors[f"{split}Features"], tensors[f"{split}Labels"] = features, labels
      feature_index[split] = [{k: r[k] for k in (
        "sampleId", "speciesId", "imageSha256", "observerGroup", "split",
      )} for r in splits[split]]
    feature_seconds = time.monotonic() - feature_start
    if state_digest(backbone.state_dict()) != backbone_before:
      raise ValueError("Frozen backbone or BatchNorm buffers changed.")
    save_file(tensors, output / "features.safetensors")
    write_json(output / "feature-index.json", {
      "featuresSha256": sha256(output / "features.safetensors"),
      "protocolSha256": sha256(output / "protocol.json"),
      "backboneSha256": backbone_before, "samples": feature_index,
      "featureExtractionSeconds": feature_seconds,
      "featureArrayBytes": sum(
        t.numel() * t.element_size() for k, t in tensors.items() if k.endswith("Features")
      ),
      "freshFromRawImages": True, "oldFeatureCacheUsed": False,
      "fixedViewsPerImage": 1, "randomImageAugmentation": False,
    })
    tensors = load_file(output / "features.safetensors")
    fitted = fit_head(head, tensors, splits, classes, output)
    if state_digest(backbone.state_dict()) != backbone_before:
      raise ValueError("Head training modified the backbone.")
    backbone.fc = head
    backbone.eval()
    save_file({k: v.detach().contiguous() for k, v in backbone.state_dict().items()},
              output / "classifier.safetensors")
    training_counts = Counter(r["speciesId"] for r in splits["train"])
    metadata = {
      **fitted, "status": "trained-offline", "completedAt": now(),
      "architecture": "resnet18", "classCount": len(classes),
      "initialization": RECIPE["weights"], "preprocessing": RECIPE["preprocessing"],
      "protocolSha256": sha256(output / "protocol.json"),
      "weightsSha256": sha256(output / "classifier.safetensors"),
      "sourceWeightsSha256": WEIGHTS_SHA, "frozenBackboneSha256": backbone_before,
      "featuresSha256": sha256(output / "features.safetensors"),
      "classes": [{**r, "trainingExamples": training_counts[r["speciesId"]],
                   "classRole": "trained-linear-target"}
                  for r in classes],
      "validation": json.loads((output / "validation.json").read_text()),
      "featureExtractionSeconds": feature_seconds,
      "elapsedSeconds": time.monotonic() - started,
      "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      "trainBackbone": False, "batchNormStatisticsChanged": False,
      "pretrainedWeightsUsed": True, "inheritedClassCount": 0,
      "externalSpeciesFallbackAllowed": False, "confidenceCalibrated": False,
      "testUsed": False, "independentFinalBenchmark": False, "productionApproved": False,
      "includesBinaryClassifier": False, "automaticStageB": False,
    }
    write_json(output / "trained-species.json", {
      "schemaVersion": "linear-head-experiment-v1",
      **{k: metadata[k] for k in (
        "status", "classCount", "classes", "weightsSha256", "protocolSha256",
        "pretrainedWeightsUsed", "inheritedClassCount", "productionApproved",
      )},
    })
    write_json(output / "model.json", metadata)
    write_json(output / "progress.json", {
      "stage": "trained-offline", "updatedAt": now(),
      **fitted, "onlineChanged": False, "automaticStageB": False,
    })
  except BaseException as exc:
    write_json(output / "failure.json", {
      "stage": "failed", "updatedAt": now(), "errorType": type(exc).__name__,
    })
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--weights", required=True, type=Path)
  args = parser.parse_args()
  train(args.data.resolve(), args.output.resolve(), args.weights.resolve())


if __name__ == "__main__":
  main()
