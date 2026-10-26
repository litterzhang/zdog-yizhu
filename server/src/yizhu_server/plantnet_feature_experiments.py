"""Feature-only Pl@ntNet head experiments with fixed validation selection and test isolation."""

import argparse
import json
import os
import random
import resource
import time
from pathlib import Path

from yizhu_server.distributed_plantnet_features import DIMENSION, WEIGHTS_SHA256
from yizhu_server.training_data import now, sha256, write_json

MODES = {"focus82", "retained496", "adapt496"}


def load_sources(base, supplements, classes):
  import torch
  from safetensors.torch import load_file

  class_ids = {r["speciesId"] for r in classes}
  if len(class_ids) != len(classes):
    raise ValueError("Duplicate target classes.")
  sources, seen_samples, image_owners, seen_views = [], set(), {}, set()
  for root in [base, *supplements]:
    index = json.loads((root / "index.json").read_text())
    if (
      index.get("status") != "features-extracted"
      or index.get("checkpointSha256") != WEIGHTS_SHA256
      or index.get("dimension") != DIMENSION or index.get("normalized") is not False
      or sha256(root / "features.safetensors") != index["featuresSha256"]
      or index.get("trainingStarted") is not False
    ):
      raise ValueError("Feature source identity or policy changed.")
    tensors = load_file(root / "features.safetensors")
    features = tensors.get("features")
    if (
      features is None or features.shape != (len(index["samples"]), DIMENSION)
      or features.dtype != torch.float32 or not torch.isfinite(features).all()
    ):
      raise ValueError("Invalid feature tensor.")
    for row in index["samples"]:
      base_sample = row.get("baseSampleId", row["sampleId"])
      view = row.get("featureView", "official-center-crop")
      owner = (base_sample, row["speciesId"])
      if row["sampleId"] in seen_samples or (base_sample, view) in seen_views:
        raise ValueError("Duplicate or out-of-scope feature row.")
      if row["speciesId"] not in class_ids:
        raise ValueError("Duplicate or out-of-scope feature row.")
      prior = image_owners.get(row["imageSha256"])
      if prior is not None and prior != owner:
        raise ValueError("Same image hash has conflicting base sample or label.")
      seen_samples.add(row["sampleId"])
      seen_views.add((base_sample, view))
      image_owners[row["imageSha256"]] = owner
    sources.append((root, index, features))
  return sources


def retained_contract(path):
  contract = json.loads(path.read_text())
  rows = contract.get("supportedSpecies")
  if (
    contract.get("solutionId") != "plantnet-300k-resnet50"
    or contract.get("status") != "pretrained" or not isinstance(rows, list)
    or len(rows) != 420
    or contract.get("sourceFiles", {}).get(
      "resnet50_weights_best_acc.tar", {},
    ).get("sha256") != WEIGHTS_SHA256
  ):
    raise ValueError("Require the pinned 420-class mapped pretrained contract.")
  rows = sorted(rows, key=lambda row: row["sourceClasses"][0]["classIndex"])
  indices = []
  for row in rows:
    refs = row.get("sourceClasses", [])
    if (
      len(refs) != 1 or type(refs[0].get("classIndex")) is not int
      or not 0 <= refs[0]["classIndex"] < 1081
    ):
      raise ValueError("Retained class mapping is ambiguous.")
    indices.append(refs[0]["classIndex"])
  if len(set(indices)) != len(indices) or len({r["speciesId"] for r in rows}) != len(rows):
    raise ValueError("Duplicate retained source row or stable species ID.")
  return rows, indices


def original_head(weights):
  import torch

  if weights.stat().st_size != 206_149_183 or sha256(weights) != WEIGHTS_SHA256:
    raise ValueError("Pinned Pl@ntNet checkpoint missing or changed.")
  checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
  state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
  if (
    not isinstance(state, dict) or state.get("fc.weight") is None
    or state["fc.weight"].shape != (1081, DIMENSION)
    or state["fc.bias"].shape != (1081,)
  ):
    raise ValueError("Unexpected original classifier head.")
  return state["fc.weight"].detach().clone(), state["fc.bias"].detach().clone()


def assemble(sources, output_classes):
  import torch

  indices = {r["speciesId"]: i for i, r in enumerate(output_classes)}
  buckets = {split: [[], []] for split in ("train", "validation", "test")}
  for _, index, features in sources:
    for i, row in enumerate(index["samples"]):
      split = row.get("split", "train")
      if split not in buckets:
        raise ValueError("Unexpected feature split.")
      buckets[split][0].append(features[i])
      buckets[split][1].append(indices[row["speciesId"]])
  result = {}
  for split, (vectors, labels) in buckets.items():
    if not vectors:
      raise ValueError(f"Missing {split} features.")
    result[split] = (torch.stack(vectors), torch.tensor(labels, dtype=torch.long))
  return result


def metrics(logits, labels, classes, *, target_ids=None):
  import torch

  if logits.shape != (len(labels), len(classes)) or not torch.isfinite(logits).all():
    raise ValueError("Invalid experiment logits.")
  ranking = logits.topk(min(5, len(classes)), dim=1).indices
  per_class = {}
  for label in labels.unique().tolist():
    mask = labels == label
    per_class[classes[label]["speciesId"]] = {
      "samples": int(mask.sum()),
      "top1": float(ranking[mask, 0].eq(labels[mask]).float().mean()),
      "top5": float(ranking[mask].eq(labels[mask, None]).any(1).float().mean()),
    }
  result = {
    "samples": len(labels),
    "top1": float(ranking[:, 0].eq(labels).float().mean()),
    "top5": float(ranking.eq(labels[:, None]).any(1).float().mean()),
    "macroTop1": sum(r["top1"] for r in per_class.values()) / len(per_class),
    "macroTop5": sum(r["top5"] for r in per_class.values()) / len(per_class),
    "loss": float(torch.nn.functional.cross_entropy(logits, labels)),
    "perClass": per_class,
  }
  if target_ids is not None:
    for name, ids in (
      ("target82", set(target_ids)),
      ("otherObserved", {r["speciesId"] for r in classes} - set(target_ids)),
    ):
      rows = [row for sid, row in per_class.items() if sid in ids]
      if not rows:
        continue
      samples = sum(row["samples"] for row in rows)
      result.setdefault("groups", {})[name] = {
        "species": len(rows), "samples": samples,
        "top1": sum(row["top1"] * row["samples"] for row in rows) / samples,
        "top5": sum(row["top5"] * row["samples"] for row in rows) / samples,
        "macroTop1": sum(row["top1"] for row in rows) / len(rows),
        "macroTop5": sum(row["top5"] for row in rows) / len(rows),
      }
  return result


def selection_key(result):
  # User explicitly prioritized Top-1. Ties preserve Top-5, then lower CE.
  return result["macroTop1"], result["macroTop5"], -result["loss"]


def train(mode, base, supplements, classes_path, contract_path, weights, output, *,
          epochs=200, patience=25, seed=20260930, threads=2,
          label_smoothing=0.0, weight_decay=.0001, feature_dropout=0.0):
  import torch
  from safetensors.torch import save_file

  if (
    mode not in MODES or output.exists() or not 1 <= epochs <= 1000
    or not 1 <= patience <= epochs or not 1 <= threads <= 8
    or not 0 <= label_smoothing <= .3 or not 0 <= weight_decay <= .1
    or not 0 <= feature_dropout <= .5
  ):
    raise ValueError("Invalid mode, output, or bounded training settings.")
  target_classes = json.loads(classes_path.read_text())["classes"]
  if len(target_classes) != 82:
    raise ValueError("This experiment requires the frozen 82-class target set.")
  retained, source_indices = retained_contract(contract_path)
  retained_by_id = {r["speciesId"]: r for r in retained}
  overlap = [r for r in target_classes if r["speciesId"] in retained_by_id]
  additions = [r for r in target_classes if r["speciesId"] not in retained_by_id]
  if len(overlap) != 6 or len(additions) != 76:
    raise ValueError("Pretrained/target overlap changed; review mappings before training.")
  original_weight, original_bias = original_head(weights)
  retained_weight = original_weight[source_indices]
  retained_bias = original_bias[source_indices]
  if mode == "focus82":
    output_classes = [{
      **row,
      "classRole": (
        "pretrained-row-initialized-and-target-trained"
        if row["speciesId"] in retained_by_id else "target-trained"
      ),
    } for row in target_classes]
    fixed_count = 0
  else:
    output_classes = [
      {
        **row,
        "classRole": (
          "pretrained-row-target-adapted"
          if mode == "adapt496" and row["speciesId"] in {r["speciesId"] for r in overlap}
          else "retained-pretrained"
        ),
      } for row in retained
    ] + [
      {**row, "classRole": "target-trained"} for row in additions
    ]
    fixed_count = len(retained)
  sources = load_sources(base, supplements, output_classes)
  splits = assemble(sources, output_classes)
  torch.set_num_threads(threads)
  torch.manual_seed(seed)
  random.seed(seed)
  target_output_indices = [
    next(i for i, item in enumerate(output_classes) if item["speciesId"] == row["speciesId"])
    for row in target_classes
  ]
  if mode == "focus82":
    trainable = torch.nn.Linear(DIMENSION, len(output_classes))
    with torch.no_grad():
      for row in overlap:
        target = next(i for i, item in enumerate(output_classes)
                      if item["speciesId"] == row["speciesId"])
        retained_index = next(i for i, item in enumerate(retained)
                              if item["speciesId"] == row["speciesId"])
        trainable.weight[target].copy_(retained_weight[retained_index])
        trainable.bias[target].copy_(retained_bias[retained_index])
  elif mode == "retained496":
    trainable = torch.nn.Linear(DIMENSION, len(additions))
  else:
    trainable = torch.nn.Linear(DIMENSION, len(target_classes))
    with torch.no_grad():
      for target_index, row in enumerate(target_classes):
        if row["speciesId"] not in retained_by_id:
          continue
        retained_index = next(i for i, item in enumerate(retained)
                              if item["speciesId"] == row["speciesId"])
        trainable.weight[target_index].copy_(retained_weight[retained_index])
        trainable.bias[target_index].copy_(retained_bias[retained_index])
  optimizer = torch.optim.AdamW(
    trainable.parameters(), lr=.001, weight_decay=weight_decay,
  )
  x_train, y_train = splits["train"]
  frequencies = torch.bincount(y_train, minlength=len(output_classes))
  observed = frequencies > 0
  if mode == "retained496":
    if not observed[:fixed_count].any() or not observed[fixed_count:].all():
      raise ValueError("Need retained replay and every additional training class.")
  elif mode == "adapt496":
    if not observed[target_output_indices].all():
      raise ValueError("Every adapted target row requires training examples.")
  elif not observed.all():
    raise ValueError("Every focus class requires training examples.")
  class_weights = torch.zeros(len(output_classes))
  class_weights[observed] = len(y_train) / (
    int(observed.sum()) * frequencies[observed].float()
  )
  generator = torch.Generator().manual_seed(seed)
  output.mkdir(parents=True)
  plan = {
    "createdAt": now(), "mode": mode, "seed": seed, "epochs": epochs,
    "patience": patience, "batchSize": 256, "optimizer": "AdamW",
    "learningRate": .001, "weightDecay": weight_decay,
    "labelSmoothing": label_smoothing,
    "featureDropout": feature_dropout,
    "selection": "highest validation macro Top-1; tie Top-5; tie lower CE",
    "classBalancedLoss": True, "classCount": len(output_classes),
    "retainedClassCount": (
      fixed_count if mode == "retained496" else fixed_count - len(overlap)
      if mode == "adapt496" else 0
    ),
    "trainedClassCount": (
      len(output_classes) - fixed_count if mode == "retained496"
      else len(target_classes) if mode == "adapt496" else len(output_classes)
    ),
    "targetClassCount": 82, "pretrainedTargetOverlap": len(overlap),
    "sourceFeatureFiles": {
      root.name: {
        "indexSha256": sha256(root / "index.json"),
        "featuresSha256": sha256(root / "features.safetensors"),
      } for root in [base, *supplements]
    },
    "pretrainedContractSha256": sha256(contract_path),
    "checkpointSha256": WEIGHTS_SHA256,
    "testUsedForSelection": False, "independentFinalBenchmark": False,
    "automaticDeployment": False,
  }
  write_json(output / "plan.json", plan)

  def logits(x):
    if mode == "focus82":
      return trainable(x)
    if mode == "retained496":
      return torch.cat((
        torch.nn.functional.linear(x, retained_weight, retained_bias), trainable(x),
      ), dim=1)
    fixed_weight = torch.cat((
      retained_weight, torch.zeros((len(additions), DIMENSION)),
    ), dim=0)
    fixed_bias = torch.cat((retained_bias, torch.zeros(len(additions))), dim=0)
    target_indices = torch.tensor(target_output_indices, dtype=torch.long)
    return torch.nn.functional.linear(x, fixed_weight, fixed_bias).index_copy(
      1, target_indices, trainable(x),
    )

  history, best_key, best_state, best_epoch, stale = [], None, None, 0, 0
  started = time.monotonic()
  for epoch in range(1, epochs + 1):
    trainable.train()
    permutation = torch.randperm(len(y_train), generator=generator)
    loss_sum = 0.0
    for batch in permutation.split(256):
      optimizer.zero_grad(set_to_none=True)
      inputs = torch.nn.functional.dropout(
        x_train[batch], p=feature_dropout, training=feature_dropout > 0,
      )
      scores = logits(inputs)
      loss = torch.nn.functional.cross_entropy(
        scores, y_train[batch], weight=class_weights,
        label_smoothing=label_smoothing,
      )
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss.")
      loss.backward()
      torch.nn.utils.clip_grad_norm_(trainable.parameters(), 5., error_if_nonfinite=True)
      optimizer.step()
      loss_sum += float(loss.detach()) * len(batch)
    trainable.eval()
    with torch.inference_mode():
      validation = metrics(
        logits(splits["validation"][0]), splits["validation"][1],
        output_classes, target_ids={r["speciesId"] for r in target_classes},
      )
    key = selection_key(validation)
    summary = {
      "epoch": epoch, "trainOnlineLoss": loss_sum / len(y_train),
      **{k: validation[k] for k in (
        "top1", "top5", "macroTop1", "macroTop5", "loss",
      )},
      "elapsedSeconds": time.monotonic() - started,
    }
    history.append(summary)
    if best_key is None or key > best_key:
      best_key, best_epoch, stale = key, epoch, 0
      best_state = {k: v.detach().clone() for k, v in trainable.state_dict().items()}
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
  trainable.load_state_dict(best_state)
  trainable.eval()
  with torch.inference_mode():
    results = {
      split: metrics(
        logits(x), y, output_classes,
        target_ids={r["speciesId"] for r in target_classes},
      )
      for split, (x, y) in splits.items()
    }
  if mode == "focus82":
    state = trainable.state_dict()
  elif mode == "retained496":
    state = {
      "weight": torch.cat((retained_weight, trainable.weight.detach()), dim=0),
      "bias": torch.cat((retained_bias, trainable.bias.detach()), dim=0),
    }
    if (
      not torch.equal(state["weight"][:fixed_count], retained_weight)
      or not torch.equal(state["bias"][:fixed_count], retained_bias)
    ):
      raise ValueError("Retained classifier rows changed.")
  else:
    state = {
      "weight": torch.cat((
        retained_weight, torch.zeros((len(additions), DIMENSION)),
      ), dim=0),
      "bias": torch.cat((retained_bias, torch.zeros(len(additions))), dim=0),
    }
    state["weight"][target_output_indices] = trainable.weight.detach()
    state["bias"][target_output_indices] = trainable.bias.detach()
    untouched = [
      i for i in range(fixed_count) if i not in set(target_output_indices)
    ]
    if (
      not torch.equal(state["weight"][untouched], retained_weight[untouched])
      or not torch.equal(state["bias"][untouched], retained_bias[untouched])
    ):
      raise ValueError("Non-target retained classifier rows changed.")
  temporary = output / "classifier.safetensors.tmp"
  save_file({k: v.contiguous() for k, v in state.items()}, temporary)
  os.replace(temporary, output / "classifier.safetensors")
  write_json(output / "results.json", {
    "createdAt": now(), "mode": mode, "classCount": len(output_classes),
    "classes": output_classes, "bestEpoch": best_epoch, "epochsRun": len(history),
    "metrics": results, "goal": {"top1": .80, "top5": .90},
    "validationGoalMet": (
      results["validation"]["macroTop1"] > .80
      and results["validation"]["macroTop5"] > .90
    ),
    "targetValidationGoalMet": (
      results["validation"]["groups"]["target82"]["macroTop1"] > .80
      and results["validation"]["groups"]["target82"]["macroTop5"] > .90
    ),
    "coverageGoalMet": len(output_classes) >= 200,
    "finalIndependentGoalMet": False,
    "weightsSha256": sha256(output / "classifier.safetensors"),
    "retainedRowsUnchanged": mode == "retained496",
    "unchangedPretrainedRows": (
      420 if mode == "retained496" else 414 if mode == "adapt496" else 0
    ),
    "testUsedForSelection": False, "productionApproved": False,
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  })
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "trained-offline", "bestEpoch": best_epoch,
    "epochsRun": len(history), "onlineChanged": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=sorted(MODES), required=True)
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--supplement", type=Path, action="append", default=[])
  parser.add_argument("--classes", type=Path, required=True)
  parser.add_argument("--contract", type=Path, required=True)
  parser.add_argument("--weights", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--patience", type=int, default=25)
  parser.add_argument("--seed", type=int, default=20260930)
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument("--label-smoothing", type=float, default=0.0)
  parser.add_argument("--weight-decay", type=float, default=.0001)
  parser.add_argument("--feature-dropout", type=float, default=0.0)
  args = parser.parse_args()
  train(
    args.mode, args.base.resolve(), [p.resolve() for p in args.supplement],
    args.classes.resolve(), args.contract.resolve(), args.weights.resolve(),
    args.output.resolve(), epochs=args.epochs, patience=args.patience,
    seed=args.seed, threads=args.threads, label_smoothing=args.label_smoothing,
    weight_decay=args.weight_decay, feature_dropout=args.feature_dropout,
  )


if __name__ == "__main__":
  main()
