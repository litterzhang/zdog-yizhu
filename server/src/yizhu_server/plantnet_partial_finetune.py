"""Fine-tune Pl@ntNet ResNet50 layer4 with target rows and old-logit distillation."""

import argparse
import hashlib
import json
import random
import resource
import time
from collections import Counter
from pathlib import Path

from yizhu_server.distributed_plantnet_features import (
  DIMENSION,
  WEIGHTS_SHA256,
  augmented_preprocess,
  preprocess,
)
from yizhu_server.recognition import read_photo
from yizhu_server.training_data import now, sha256, write_json


def load_feature_map(roots):
  from safetensors.torch import load_file

  result = {}
  for root in roots:
    index = json.loads((root / "index.json").read_text())
    if (
      index.get("status") != "features-extracted"
      or index.get("checkpointSha256") != WEIGHTS_SHA256
      or index.get("dimension") != DIMENSION or index.get("normalized") is not False
      or index.get("viewSeed") is not None
      or sha256(root / "features.safetensors") != index["featuresSha256"]
    ):
      raise ValueError("Require original center-view teacher features.")
    vectors = load_file(root / "features.safetensors")["features"]
    for row in index["samples"]:
      if row["sampleId"] in result:
        raise ValueError("Duplicate teacher feature sample.")
      result[row["sampleId"]] = vectors[row["row"]]
  return result


def collection_rows(root, accepted_path):
  accepted = json.loads(accepted_path.read_text())
  wanted = {r["sampleId"]: r for r in accepted["samples"]}
  rows = []
  for path in sorted((root / "samples").glob("*.json")):
    row = json.loads(path.read_text())["samples"][0]
    expected = wanted.get(row["sampleId"])
    if expected is None:
      continue
    if (
      any(row[k] != expected[k] for k in (
        "speciesId", "imageSha256", "observerGroup", "photoId", "observationId",
      ))
      or row["split"] != "train" or sha256(root / row["path"]) != row["imageSha256"]
    ):
      raise ValueError("Accepted collection row or image changed.")
    rows.append(({**row, "sourceRoot": str(root)}, root / row["path"]))
  if len(rows) != len(wanted):
    raise ValueError("Not all accepted collection rows are available.")
  return rows


def digest_state(state):
  digest = hashlib.sha256()
  for name, value in sorted(state.items()):
    digest.update(name.encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def set_training_mode(model):
  import torch

  model.eval()
  model.layer4.train()
  for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.eval()


def metrics(logits, labels, classes):
  import torch

  ranking = logits.topk(5, 1).indices
  per = {}
  for label in labels.unique().tolist():
    mask = labels == label
    per[classes[label]["speciesId"]] = {
      "samples": int(mask.sum()),
      "top1": float(ranking[mask, 0].eq(labels[mask]).float().mean()),
      "top5": float(ranking[mask].eq(labels[mask, None]).any(1).float().mean()),
    }
  return {
    "samples": len(labels),
    "top1": float(ranking[:, 0].eq(labels).float().mean()),
    "top5": float(ranking.eq(labels[:, None]).any(1).float().mean()),
    "macroTop1": sum(r["top1"] for r in per.values()) / len(per),
    "macroTop5": sum(r["top5"] for r in per.values()) / len(per),
    "loss": float(torch.nn.functional.cross_entropy(logits, labels)),
    "perClass": per,
  }


def train(base_data, collections, accepted, teacher_sources, start_model, weights, output, *,
          epochs=10, patience=3, threads=4, seed=20260930, distill_weight=.25):
  import torch
  from safetensors.torch import load_file, save_file
  from torch.utils.data import DataLoader, Dataset
  from torchvision.models import resnet50

  if (
    output.exists() or len(collections) != len(accepted) or not 1 <= epochs <= 30
    or not 1 <= patience <= epochs or not 0 <= distill_weight <= 2
  ):
    raise ValueError("Invalid bounded partial-finetune plan.")
  training = json.loads((base_data / "train-validation.json").read_text())
  target_classes = training["classes"]
  target_ids = {r["speciesId"] for r in target_classes}
  start_results = json.loads((start_model / "results.json").read_text())
  classes = start_results["classes"]
  if (
    start_results.get("mode") != "adapt496" or len(classes) != 496
    or start_results.get("validationGoalMet") is not True
    or sha256(start_model / "classifier.safetensors") != start_results["weightsSha256"]
  ):
    raise ValueError("Require the selected 496-class adapted starting head.")
  class_index = {r["speciesId"]: i for i, r in enumerate(classes)}
  if not target_ids <= set(class_index):
    raise ValueError("Starting model misses target classes.")
  target_indices = [class_index[r["speciesId"]] for r in target_classes]
  non_target_indices = [i for i in range(len(classes)) if i not in set(target_indices)]
  head = load_file(start_model / "classifier.safetensors")
  teacher_features = load_feature_map(teacher_sources)

  rows = []
  for row in training["samples"]:
    path = base_data / row["path"]
    if sha256(path) != row["imageSha256"]:
      raise ValueError("Base image changed.")
    rows.append((row, path))
  for root, manifest in zip(collections, accepted, strict=True):
    rows.extend(collection_rows(root, manifest))
  seen = set()
  for row, _ in rows:
    if row["sampleId"] in seen or row["sampleId"] not in teacher_features:
      raise ValueError("Duplicate row or missing teacher feature.")
    seen.add(row["sampleId"])
  train_rows = [(r, p) for r, p in rows if r["split"] == "train"]
  val_rows = [(r, p) for r, p in rows if r["split"] == "validation"]
  if len(val_rows) != 1230 or any(r["split"] != "validation" for r, _ in val_rows):
    raise ValueError("Validation split changed.")
  train_transform, val_transform = augmented_preprocess(), preprocess()

  class Photos(Dataset):
    def __init__(self, entries, transform, include_teacher):
      self.entries, self.transform, self.include_teacher = entries, transform, include_teacher

    def __len__(self):
      return len(self.entries)

    def __getitem__(self, index):
      row, path = self.entries[index]
      with read_photo(path) as image:
        pixels = self.transform(image)
      teacher = (
        teacher_features[row["sampleId"]] if self.include_teacher
        else torch.empty(0)
      )
      return pixels, class_index[row["speciesId"]], teacher

  torch.set_num_threads(threads)
  torch.set_num_interop_threads(1)
  torch.manual_seed(seed)
  random.seed(seed)
  checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
  state = checkpoint.get("model")
  if (
    weights.stat().st_size != 206_149_183 or sha256(weights) != WEIGHTS_SHA256
    or not isinstance(state, dict) or state["fc.weight"].shape != (1081, DIMENSION)
  ):
    raise ValueError("Pinned Pl@ntNet checkpoint changed.")
  model = resnet50(weights=None, num_classes=1081)
  model.load_state_dict(state, strict=True)
  model.fc = torch.nn.Identity()
  model.requires_grad_(False)
  model.layer4.requires_grad_(True)
  for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.requires_grad_(False)
  target_head = torch.nn.Linear(DIMENSION, len(target_classes))
  with torch.no_grad():
    target_head.weight.copy_(head["weight"][target_indices])
    target_head.bias.copy_(head["bias"][target_indices])
  fixed_weight, fixed_bias = head["weight"].clone(), head["bias"].clone()
  frozen_before = digest_state({
    k: v for k, v in model.state_dict().items() if not k.startswith("layer4.")
  })
  bn_before = digest_state({
    k: v for k, v in model.state_dict().items()
    if "bn" in k or "downsample.1" in k
  })
  optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": target_head.parameters(), "lr": 5e-4},
  ], weight_decay=1e-4)
  frequencies = Counter(class_index[r["speciesId"]] for r, _ in train_rows)
  class_weights = torch.zeros(len(classes))
  for label, count in frequencies.items():
    class_weights[label] = len(train_rows) / (len(frequencies) * count)
  generator = torch.Generator().manual_seed(seed)
  train_loader = DataLoader(
    Photos(train_rows, train_transform, True), batch_size=16, shuffle=True,
    generator=generator, num_workers=0,
  )
  val_loader = DataLoader(
    Photos(val_rows, val_transform, False), batch_size=16, shuffle=False, num_workers=0,
  )
  target_tensor = torch.tensor(target_indices, dtype=torch.long)
  old_tensor = torch.tensor(non_target_indices, dtype=torch.long)

  def features(images, *, grad):
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context:
      with torch.no_grad():
        value = model.conv1(images)
        value = model.bn1(value)
        value = model.relu(value)
        value = model.maxpool(value)
        value = model.layer1(value)
        value = model.layer2(value)
        value = model.layer3(value)
      value = model.layer4(value)
      value = model.avgpool(value)
      return torch.flatten(value, 1)

  def logits(vector):
    base = torch.nn.functional.linear(vector, fixed_weight, fixed_bias)
    return base.index_copy(1, target_tensor, target_head(vector))

  output.mkdir(parents=True)
  write_json(output / "plan.json", {
    "createdAt": now(), "architecture": "plantnet-resnet50-layer4-target82-adapter",
    "classCount": len(classes), "targetTrainableRows": len(target_indices),
    "unchangedHeadRows": len(non_target_indices), "trainImages": len(train_rows),
    "validationImages": len(val_rows), "epochs": epochs, "patience": patience,
    "seed": seed, "batchSize": 16, "layer4LearningRate": 1e-5,
    "targetHeadLearningRate": 5e-4, "weightDecay": 1e-4,
    "labelSmoothing": .1, "distillation": {
      "weight": distill_weight, "temperature": 2.0,
      "scope": "414 non-target pretrained logits against frozen center-view teacher features",
    },
    "augmentation": "RandomResizedCrop224+flip+ColorJitter",
    "selection": "highest validation macro Top-1; tie Top-5; tie lower CE",
    "startWeightsSha256": start_results["weightsSha256"],
    "testUsedForSelection": False, "automaticDeployment": False,
  })
  history, best_key, best, best_epoch, stale = [], None, None, 0, 0
  started = time.monotonic()
  for epoch in range(1, epochs + 1):
    set_training_mode(model)
    target_head.train()
    total_loss, seen_count = 0.0, 0
    for images, labels, teacher_vector in train_loader:
      optimizer.zero_grad(set_to_none=True)
      vector = features(images, grad=True)
      scores = logits(vector)
      ce = torch.nn.functional.cross_entropy(
        scores, labels, weight=class_weights, label_smoothing=.1,
      )
      with torch.no_grad():
        teacher_old = torch.nn.functional.linear(
          teacher_vector, fixed_weight[old_tensor], fixed_bias[old_tensor],
        )
      student_old = torch.nn.functional.linear(
        vector, fixed_weight[old_tensor], fixed_bias[old_tensor],
      )
      temperature = 2.0
      distill = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_old / temperature, dim=1),
        torch.nn.functional.softmax(teacher_old / temperature, dim=1),
        reduction="batchmean",
      ) * temperature**2
      loss = ce + distill_weight * distill
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite partial-finetune loss.")
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        [*model.layer4.parameters(), *target_head.parameters()],
        5., error_if_nonfinite=True,
      )
      optimizer.step()
      total_loss += float(loss.detach()) * len(labels)
      seen_count += len(labels)
    if (
      digest_state({k: v for k, v in model.state_dict().items()
                    if not k.startswith("layer4.")}) != frozen_before
      or digest_state({k: v for k, v in model.state_dict().items()
                       if "bn" in k or "downsample.1" in k}) != bn_before
    ):
      raise ValueError("Frozen prefix or BatchNorm changed.")
    model.eval()
    target_head.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
      for images, labels, _ in val_loader:
        all_logits.append(logits(features(images, grad=False)))
        all_labels.append(labels)
    result = metrics(torch.cat(all_logits), torch.cat(all_labels), classes)
    key = result["macroTop1"], result["macroTop5"], -result["loss"]
    summary = {
      "epoch": epoch, "trainLoss": total_loss / seen_count,
      **{k: result[k] for k in ("top1", "top5", "macroTop1", "macroTop5", "loss")},
      "elapsedSeconds": time.monotonic() - started,
    }
    history.append(summary)
    if best_key is None or key > best_key:
      best_key, best_epoch, stale = key, epoch, 0
      best = {
        "layer4": {k: v.detach().clone() for k, v in model.layer4.state_dict().items()},
        "targetHead": {k: v.detach().clone() for k, v in target_head.state_dict().items()},
        "validation": result,
      }
    else:
      stale += 1
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      "updatedAt": now(), "stage": "training", "bestEpoch": best_epoch,
      "staleEpochs": stale, **summary,
    })
    print(json.dumps(summary), flush=True)
    if stale >= patience:
      break
  model.layer4.load_state_dict(best["layer4"])
  target_head.load_state_dict(best["targetHead"])
  save_file(
    {f"layer4.{k}": v.contiguous() for k, v in model.layer4.state_dict().items()},
    output / "layer4.safetensors",
  )
  final_head = {"weight": fixed_weight.clone(), "bias": fixed_bias.clone()}
  final_head["weight"][target_indices] = target_head.weight.detach()
  final_head["bias"][target_indices] = target_head.bias.detach()
  save_file({k: v.contiguous() for k, v in final_head.items()}, output / "classifier.safetensors")
  write_json(output / "results.json", {
    "createdAt": now(), "classCount": len(classes), "classes": classes,
    "bestEpoch": best_epoch, "epochsRun": len(history),
    "validation": best["validation"],
    "validationGoalMet": (
      best["validation"]["macroTop1"] > .8 and best["validation"]["macroTop5"] > .9
    ),
    "coverageGoalMet": True, "finalIndependentGoalMet": False,
    "layer4Sha256": sha256(output / "layer4.safetensors"),
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
  parser.add_argument("--base-data", type=Path, required=True)
  parser.add_argument("--collection", type=Path, action="append", default=[])
  parser.add_argument("--accepted", type=Path, action="append", default=[])
  parser.add_argument("--teacher-source", type=Path, action="append", required=True)
  parser.add_argument("--start-model", type=Path, required=True)
  parser.add_argument("--weights", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--patience", type=int, default=3)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--seed", type=int, default=20260930)
  parser.add_argument("--distill-weight", type=float, default=.25)
  args = parser.parse_args()
  train(
    args.base_data.resolve(), [p.resolve() for p in args.collection],
    [p.resolve() for p in args.accepted], [p.resolve() for p in args.teacher_source],
    args.start_model.resolve(), args.weights.resolve(), args.output.resolve(),
    epochs=args.epochs, patience=args.patience, threads=args.threads,
    seed=args.seed, distill_weight=args.distill_weight,
  )


if __name__ == "__main__":
  main()
