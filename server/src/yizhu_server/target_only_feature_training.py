"""Train a target-list-only feature head; never add labels outside the reviewed target list."""

import argparse
import json
import os
import random
import resource
import time
from collections import Counter
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def evaluate(logits, expected_ids, output_classes, requested_classes):
  output_ids = [r["speciesId"] for r in output_classes]
  ranking = logits.topk(min(5, len(output_classes)), 1).indices
  predictions = [[output_ids[i] for i in row] for row in ranking.tolist()]
  per = {}
  for row in requested_classes:
    sid = row["speciesId"]
    indices = [i for i, expected in enumerate(expected_ids) if expected == sid]
    if not indices:
      continue
    per[sid] = {
      "samples": len(indices),
      "supported": sid in set(output_ids),
      "top1": sum(predictions[i][0] == sid for i in indices) / len(indices),
      "top5": sum(sid in predictions[i] for i in indices) / len(indices),
    }
  n = len(expected_ids)
  return {
    "samples": n, "evaluatedSpecies": len(per),
    "supportedEvaluatedSpecies": sum(r["supported"] for r in per.values()),
    "top1": sum(predictions[i][0] == sid for i, sid in enumerate(expected_ids)) / n,
    "top5": sum(sid in predictions[i] for i, sid in enumerate(expected_ids)) / n,
    "macroTop1": sum(r["top1"] for r in per.values()) / len(per),
    "macroTop5": sum(r["top5"] for r in per.values()) / len(per),
    "perClass": per,
  }


def train(features_root, classes_path, contract_path, weights, output, *, epochs=200,
          patience=25, seed=20260930, threads=2, label_smoothing=.1, dropout=.1,
          feature_transform="none", class_weighting="inverse", learning_rate=.001,
          weight_decay=.0001, batch_size=256, initialization=None,
          target_initialization=None, freeze_target_initialized=False):
  import torch
  from safetensors.torch import load_file, save_file

  # Set BLAS/OpenMP limits before z-score/std or normalization kernels run.
  # Setting this only after feature transforms allowed MKL to create an
  # unbounded OpenMP team and caused a reproducible SIGILL/core dump.
  torch.set_num_threads(threads)
  torch.set_num_interop_threads(1)
  if output.exists():
    raise ValueError("Use a new target-only model directory.")
  document = json.loads(classes_path.read_text())
  requested = document["classes"]
  requested_count = len(requested)
  if requested_count < 2 or len({row["speciesId"] for row in requested}) != requested_count:
    raise ValueError("Require a reviewed target list with unique species IDs.")
  index = json.loads((features_root / "index.json").read_text())
  vectors = load_file(features_root / "features.safetensors")["features"]
  dimension = index.get("dimension")
  if (
    index.get("status") != "features-extracted"
    or sha256(features_root / "features.safetensors") != index["featuresSha256"]
    or type(dimension) is not int or not 128 <= dimension <= 8192
    or vectors.shape != (len(index["samples"]), dimension)
  ):
    raise ValueError("Feature source changed.")
  counts = Counter(
    (r["speciesId"], r["split"]) for r in index["samples"]
  )
  supported = [r for r in requested if counts[(r["speciesId"], "train")] > 0]
  supported_ids = {r["speciesId"] for r in supported}
  if len(supported) > requested_count or len(supported) * 2 < requested_count:
    raise ValueError("Invalid target-only trained class coverage.")
  class_index = {r["speciesId"]: i for i, r in enumerate(supported)}
  train_rows = [
    (i, r) for i, r in enumerate(index["samples"])
    if r["split"] == "train" and r["speciesId"] in supported_ids
  ]
  train_vectors = torch.stack([vectors[i] for i, _ in train_rows])
  transform_state = {}
  if feature_transform == "l2":
    vectors = torch.nn.functional.normalize(vectors, dim=1)
  elif feature_transform == "center-l2":
    mean = train_vectors.mean(0)
    vectors = torch.nn.functional.normalize(vectors - mean, dim=1)
    transform_state["feature_mean"] = mean
  elif feature_transform == "zscore":
    mean = train_vectors.mean(0)
    std = train_vectors.std(0, unbiased=False).clamp_min(1e-6)
    vectors = (vectors - mean) / std
    transform_state.update({"feature_mean": mean, "feature_std": std})
  elif feature_transform != "none":
    raise ValueError("Unsupported feature transform.")
  splits, evaluation = {}, {}
  for split in ("train", "validation", "test"):
    rows = [(i, r) for i, r in enumerate(index["samples"]) if r["split"] == split]
    evaluation[split] = (
      torch.stack([vectors[i] for i, _ in rows]),
      [r["speciesId"] for _, r in rows],
    )
    selected = [(i, r) for i, r in rows if r["speciesId"] in supported_ids]
    splits[split] = (
      torch.stack([vectors[i] for i, _ in selected]),
      torch.tensor([class_index[r["speciesId"]] for _, r in selected]),
      [r["speciesId"] for _, r in selected],
    )
  retained, checkpoint = {}, None
  if bool(contract_path) != bool(weights):
    raise ValueError("Pretrained classifier contract and weights must be provided together.")
  if contract_path:
    contract = json.loads(contract_path.read_text())
    retained = {r["speciesId"]: r for r in contract["supportedSpecies"]}
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    if (
      checkpoint["fc.weight"].shape[1] != dimension
      or sha256(weights) != index["checkpointSha256"]
    ):
      raise ValueError("Pretrained classifier identity changed.")
  torch.manual_seed(seed)
  random.seed(seed)
  head = torch.nn.Linear(dimension, len(supported))
  initialized, text_initialized = [], []
  target_initialized_ids = set()
  with torch.no_grad():
    for i, row in enumerate(supported):
      source = retained.get(row["speciesId"])
      if source is None:
        continue
      ref = source["sourceClasses"][0]
      head.weight[i].copy_(checkpoint["fc.weight"][ref["classIndex"]])
      head.bias[i].copy_(checkpoint["fc.bias"][ref["classIndex"]])
      initialized.append(row["speciesId"])
  initialization_evidence = None
  target_initialization_evidence = None
  if initialization and target_initialization:
    raise ValueError("Use text or target classifier initialization, not both.")
  if target_initialization:
    target_metadata = json.loads((target_initialization / "model.metadata.json").read_text())
    target_state = load_file(target_initialization / "classifier.safetensors")
    prior_plants = target_metadata.get("plantSet", {}).get("plants", [])
    prior_by_id = {row["id"]: row["classIndex"] for row in prior_plants}
    if (
      target_metadata.get("status") not in {
        "trained-offline-development-candidate", "validated-independent",
      }
      or target_metadata.get("encoder", {}).get("model") != index.get("architecture")
      or target_metadata.get("encoder", {}).get("weightsSha256")
         != index.get("checkpointSha256")
      or target_state.get("weight") is None or target_state.get("bias") is None
      or target_state["weight"].shape[1] != dimension
      or target_state["weight"].shape[0] != len(prior_plants)
      or target_state["bias"].shape != (len(prior_plants),)
      or sha256(target_initialization / "classifier.safetensors")
         != target_metadata.get("classifier", {}).get("sha256")
      or len(prior_by_id) != len(prior_plants)
    ):
      raise ValueError("Target classifier initialization differs from the feature space.")
    with torch.no_grad():
      for i, row in enumerate(supported):
        prior_index = prior_by_id.get(row["speciesId"])
        if prior_index is None:
          continue
        head.weight[i].copy_(target_state["weight"][prior_index])
        head.bias[i].copy_(target_state["bias"][prior_index])
        initialized.append(row["speciesId"])
        target_initialized_ids.add(row["speciesId"])
    target_initialization_evidence = {
      "path": str(target_initialization),
      "metadataSha256": sha256(target_initialization / "model.metadata.json"),
      "classifierSha256": sha256(target_initialization / "classifier.safetensors"),
      "initializedSpecies": len(initialized),
    }
  if initialization:
    initialization_index = json.loads((initialization / "index.json").read_text())
    initialization_state = load_file(initialization / "classifier.safetensors")
    initialization_classes = initialization_index.get("classes", [])
    if (
      initialization_index.get("status") != "text-prototypes-extracted"
      or initialization_index.get("architecture") != "imageomics/bioclip-2"
      or initialization_index.get("checkpointSha256") != index["checkpointSha256"]
      or initialization_index.get("dimension") != dimension
      or sha256(initialization / "classifier.safetensors")
         != initialization_index.get("weightsSha256")
      or [row["speciesId"] for row in initialization_classes]
         != [row["speciesId"] for row in requested]
      or initialization_state["weight"].shape != (requested_count, dimension)
      or initialization_state["bias"].shape != (requested_count,)
    ):
      raise ValueError("BioCLIP text initialization differs from the target feature space.")
    requested_index = {
      row["speciesId"]: row["classIndex"] for row in initialization_classes
    }
    with torch.no_grad():
      for i, row in enumerate(supported):
        source_index = requested_index[row["speciesId"]]
        head.weight[i].copy_(initialization_state["weight"][source_index])
        head.bias[i].copy_(initialization_state["bias"][source_index])
        text_initialized.append(row["speciesId"])
    initialization_evidence = {
      "path": str(initialization),
      "indexSha256": sha256(initialization / "index.json"),
      "weightsSha256": sha256(initialization / "classifier.safetensors"),
    }
  frequencies = torch.bincount(splits["train"][1], minlength=len(supported))
  if class_weighting == "inverse":
    weights_per_class = len(splits["train"][1]) / (
      len(supported) * frequencies.float()
    )
  elif class_weighting == "none":
    weights_per_class = None
  else:
    raise ValueError("Unsupported class weighting.")
  optimizer = torch.optim.AdamW(
    head.parameters(), lr=learning_rate, weight_decay=weight_decay,
  )
  generator = torch.Generator().manual_seed(seed)
  output.mkdir(parents=True)
  write_json(output / "plan.json", {
    "createdAt": now(), "requestedClassCount": requested_count,
    "trainedClassCount": len(supported),
    "missingTrainingClassCount": requested_count - len(supported),
    "featureArchitecture": index.get("architecture"),
    "featureDimension": dimension,
    "pretrainedRowsInitialized": len(initialized), "pretrainedSpeciesIds": initialized,
    "textPrototypeRowsInitialized": len(text_initialized),
    "textPrototypeSpeciesIds": text_initialized,
    "textPrototypeInitialization": initialization_evidence,
    "targetClassifierInitialization": target_initialization_evidence,
    "freezeTargetInitializedRows": freeze_target_initialized,
    "novelSpeciesIds": [
      row["speciesId"] for row in supported
      if row["speciesId"] not in target_initialized_ids
    ],
    "epochs": epochs, "patience": patience, "batchSize": batch_size,
    "optimizer": "AdamW", "learningRate": learning_rate,
    "weightDecay": weight_decay, "classWeighting": class_weighting,
    "featureTransform": feature_transform,
    "labelSmoothing": label_smoothing, "featureDropout": dropout,
    "selection": "requested-list validation macro Top-1; tie Top-5; tie supported CE",
    "testUsedForSelection": False, "automaticDeployment": False,
  })
  x_train, y_train, _ = splits["train"]
  best_key, best, best_epoch, stale, history = None, None, 0, 0, []
  started = time.monotonic()
  for epoch in range(1, epochs + 1):
    head.train()
    permutation = torch.randperm(len(y_train), generator=generator)
    for batch in permutation.split(batch_size):
      optimizer.zero_grad(set_to_none=True)
      logits = head(torch.nn.functional.dropout(
        x_train[batch], p=dropout, training=dropout > 0,
      ))
      loss = torch.nn.functional.cross_entropy(
        logits, y_train[batch], weight=weights_per_class,
        label_smoothing=label_smoothing,
      )
      loss.backward()
      if freeze_target_initialized and target_initialized_ids:
        frozen = torch.tensor([
          index for index, row in enumerate(supported)
          if row["speciesId"] in target_initialized_ids
        ], dtype=torch.long)
        head.weight.grad[frozen] = 0
        head.bias.grad[frozen] = 0
      torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
      optimizer.step()
    head.eval()
    with torch.inference_mode():
      x_val_all, ids_val_all = evaluation["validation"]
      result = evaluate(head(x_val_all), ids_val_all, supported, requested)
      x_val, y_val, _ = splits["validation"]
      supported_loss = float(torch.nn.functional.cross_entropy(head(x_val), y_val))
    novel_ids = {
      row["speciesId"] for row in supported
      if row["speciesId"] not in target_initialized_ids
    }
    novel_rows = [result["perClass"][sid] for sid in novel_ids if sid in result["perClass"]]
    novel_top1 = (
      sum(row["top1"] for row in novel_rows) / len(novel_rows) if novel_rows else None
    )
    novel_top5 = (
      sum(row["top5"] for row in novel_rows) / len(novel_rows) if novel_rows else None
    )
    key = (
      novel_top1 if freeze_target_initialized and novel_top1 is not None else result["macroTop1"],
      novel_top5 if freeze_target_initialized and novel_top5 is not None else result["macroTop5"],
      result["macroTop1"], result["macroTop5"], -supported_loss,
    )
    summary = {
      "epoch": epoch, **{k: result[k] for k in (
        "samples", "evaluatedSpecies", "supportedEvaluatedSpecies",
        "top1", "top5", "macroTop1", "macroTop5",
      )},
      "supportedLoss": supported_loss, "novelMacroTop1": novel_top1,
      "novelMacroTop5": novel_top5, "elapsedSeconds": time.monotonic() - started,
    }
    history.append(summary)
    if best_key is None or key > best_key:
      best_key, best_epoch, stale = key, epoch, 0
      best = {k: v.detach().clone() for k, v in head.state_dict().items()}
    else:
      stale += 1
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      "updatedAt": now(), "stage": "training", "bestEpoch": best_epoch,
      "staleEpochs": stale, **summary,
    })
    if epoch == 1 or epoch % 10 == 0:
      print(json.dumps(summary), flush=True)
    if stale >= patience:
      break
  head.load_state_dict(best)
  head.eval()
  metrics = {}
  with torch.inference_mode():
    for split, (x, ids) in evaluation.items():
      metrics[split] = evaluate(head(x), ids, supported, requested)
  temporary = output / "classifier.safetensors.tmp"
  save_file({
    **{k: v.contiguous() for k, v in head.state_dict().items()},
    **{k: v.contiguous() for k, v in transform_state.items()},
  }, temporary)
  os.replace(temporary, output / "classifier.safetensors")
  write_json(output / "results.json", {
    "createdAt": now(), "requestedClassCount": requested_count,
    "classCount": len(supported),
    "classes": [{**r, "classRole": "target-trained"} for r in supported],
    "unsupportedClasses": [r for r in requested if r["speciesId"] not in supported_ids],
    "bestEpoch": best_epoch, "epochsRun": len(history), "metrics": metrics,
    "validationGoalMet": (
      len(supported) == requested_count and metrics["validation"]["macroTop1"] > .8
      and metrics["validation"]["macroTop5"] > .9
    ),
    "coverageGoalMet": len(supported) == requested_count,
    "finalIndependentGoalMet": False,
    "weightsSha256": sha256(output / "classifier.safetensors"),
    "testUsedForSelection": False, "productionApproved": False,
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  })
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "trained-offline",
    "bestEpoch": best_epoch, "epochsRun": len(history), "onlineChanged": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--features", type=Path, required=True)
  parser.add_argument("--classes", type=Path, required=True)
  parser.add_argument("--contract", type=Path)
  parser.add_argument("--weights", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--patience", type=int, default=25)
  parser.add_argument("--seed", type=int, default=20260930)
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument(
    "--feature-transform", choices=("none", "l2", "center-l2", "zscore"),
    default="none",
  )
  parser.add_argument("--class-weighting", choices=("inverse", "none"), default="inverse")
  parser.add_argument("--label-smoothing", type=float, default=.1)
  parser.add_argument("--dropout", type=float, default=.1)
  parser.add_argument("--learning-rate", type=float, default=.001)
  parser.add_argument("--weight-decay", type=float, default=.0001)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--initialization", type=Path)
  parser.add_argument("--target-initialization", type=Path)
  parser.add_argument("--freeze-target-initialized", action="store_true")
  args = parser.parse_args()
  train(
    args.features.resolve(), args.classes.resolve(),
    args.contract.resolve() if args.contract else None,
    args.weights.resolve() if args.weights else None,
    args.output.resolve(), epochs=args.epochs,
    patience=args.patience, seed=args.seed, threads=args.threads,
    feature_transform=args.feature_transform, class_weighting=args.class_weighting,
    label_smoothing=args.label_smoothing, dropout=args.dropout,
    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
    batch_size=args.batch_size,
    initialization=args.initialization.resolve() if args.initialization else None,
    target_initialization=(
      args.target_initialization.resolve() if args.target_initialization else None
    ),
    freeze_target_initialized=args.freeze_target_initialized,
  )


if __name__ == "__main__":
  main()
