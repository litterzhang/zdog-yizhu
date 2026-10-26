"""Train full image classifiers from random initialization; offline and no deployment."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from yizhu_server.random_init_classifier import build_random_classifier
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.trained_species_contract import build_trained_contract
from yizhu_server.training_data import now, sha256, write_json


def train(data: Path, output: Path, *, epochs=30, batch_size=16, threads=2, image_size=160):
  import torch
  from PIL import Image
  from safetensors.torch import load_file, save_file
  from torch.utils.data import DataLoader, Dataset
  from torchvision import transforms

  if output.exists() or not 1 <= epochs <= 300 or not 2 <= batch_size <= 128:
    raise ValueError("Use a new output directory and bounded training settings.")
  manifest_path = data / "train-validation.json"
  manifest = json.loads(manifest_path.read_text())
  if (
    manifest.get("initialization") != "random"
    or manifest.get("pretrainedWeightsAllowed") is not False
    or manifest.get("pretrainedFeaturesAllowed") is not False
    or manifest.get("testImagesIncluded") is not False
    or manifest.get("automaticDeployment") is not False
  ):
    raise ValueError("Require an explicit clean-start train/validation dataset.")
  classes = manifest["classes"]
  indices = {row["speciesId"]: i for i, row in enumerate(classes)}
  if len(indices) != len(classes) or not 2 <= len(classes) <= 4096:
    raise ValueError("Invalid classifier labels.")
  splits = {"train": [], "validation": []}
  counts = {sid: 0 for sid in indices}
  seen = set()
  for row in manifest["samples"]:
    if (
      row["split"] not in splits or row["speciesId"] not in indices
      or row["license"] not in {"cc0", "cc-by"} or not row.get("attribution")
      or row["scientificName"] != classes[indices[row["speciesId"]]]["scientificName"]
      or row["imageSha256"] in seen
    ):
      raise ValueError("Unlicensed, conflicting, duplicate, or test sample in training input.")
    path = (data / row["path"]).resolve()
    if not path.is_relative_to((data / "images").resolve()) or sha256(path) != row["imageSha256"]:
      raise ValueError("Raw image path or checksum changed.")
    seen.add(row["imageSha256"])
    splits[row["split"]].append(row)
    if row["split"] == "train":
      counts[row["speciesId"]] += 1
  if (
    any(count < 1 for count in counts.values())
    or {r["speciesId"] for r in splits["validation"]} != set(indices)
  ):
    raise ValueError("Every output species needs own training and validation images.")
  seed = manifest["seed"]
  torch.set_num_threads(threads)
  torch.manual_seed(seed)
  random.seed(seed)
  model = build_random_classifier(len(classes), architecture="resnet18", seed=seed)
  initial_stem = model.conv1.weight.detach().clone()
  augment = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(.7, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.02),
    transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
  ])
  validation_transform = transforms.Compose([
    transforms.Resize(image_size + 24), transforms.CenterCrop(image_size),
    transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
  ])

  class Images(Dataset):
    def __init__(self, rows, transform):
      self.rows, self.transform = rows, transform

    def __len__(self):
      return len(self.rows)

    def __getitem__(self, index):
      row = self.rows[index]
      with Image.open(data / row["path"]) as image:
        tensor = self.transform(image.convert("RGB"))
      return tensor, indices[row["speciesId"]]

  generator = torch.Generator().manual_seed(seed)
  train_loader = DataLoader(
    Images(splits["train"], augment), batch_size=batch_size, shuffle=True,
    num_workers=0, generator=generator, drop_last=False,
  )
  validation_loader = DataLoader(
    Images(splits["validation"], validation_transform),
    batch_size=batch_size, shuffle=False, num_workers=0,
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
  output.mkdir(parents=True)
  plan = {
    "createdAt": now(), "manifestSha256": sha256(manifest_path),
    "architecture": "resnet18", "initialization": "random",
    "pretrainedWeightsUsed": False, "pretrainedFeaturesUsed": False,
    "inheritedClassCount": 0, "trainBackbone": True,
    "classCount": len(classes), "epochs": epochs, "batchSize": batch_size,
    "threads": threads, "imageSize": image_size, "seed": seed,
    "trainImages": len(splits["train"]), "validationImages": len(splits["validation"]),
    "optimizer": "AdamW", "learningRate": .001, "weightDecay": .01,
    "selection": "lowest validation cross entropy; patience 10",
    "testUsedForSelection": False, "automaticDeployment": False,
  }
  write_json(output / "plan.json", plan)
  history, steps, stale, best_loss, best_epoch = [], 0, 0, float("inf"), 0
  started = time.monotonic()
  for epoch in range(1, epochs + 1):
    model.train()
    train_loss, train_total = 0., 0
    epoch_examples = {sid: 0 for sid in indices}
    for batch_index, (images, labels) in enumerate(train_loader, 1):
      optimizer.zero_grad(set_to_none=True)
      logits = model(images)
      loss = torch.nn.functional.cross_entropy(logits, labels)
      if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss.")
      loss.backward()
      if model.conv1.weight.grad is None:
        raise ValueError("Backbone did not participate in training.")
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
      optimizer.step()
      steps += 1
      train_loss += float(loss.detach()) * len(labels)
      train_total += len(labels)
      for label, count in zip(*labels.unique(return_counts=True), strict=True):
        epoch_examples[classes[int(label)]["speciesId"]] += int(count)
      if batch_index == 1 or batch_index % 10 == 0:
        progress = {
          "updatedAt": now(), "stage": "training", "epoch": epoch, "maxEpochs": epochs,
          "batch": batch_index, "batches": len(train_loader),
          "optimizerSteps": steps, "trainLoss": train_loss / train_total,
          "classCount": len(classes), "initialization": "random",
          "elapsedSeconds": time.monotonic() - started,
        }
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if epoch_examples != counts:
      raise ValueError("Not every planned training image was consumed exactly once this epoch.")
    model.eval()
    val_loss, val_total, top1, top5 = 0., 0, 0, 0
    with torch.inference_mode():
      for images, labels in validation_loader:
        logits = model(images)
        val_loss += float(torch.nn.functional.cross_entropy(
          logits, labels, reduction="sum",
        ))
        ranking = logits.topk(min(5, len(classes)), dim=1).indices
        top1 += int(ranking[:, 0].eq(labels).sum())
        top5 += int(ranking.eq(labels[:, None]).any(1).sum())
        val_total += len(labels)
    score = val_loss / val_total
    if not math.isfinite(score):
      raise ValueError("Nonfinite validation score.")
    summary = {
      "epoch": epoch, "trainLoss": train_loss / train_total,
      "validationLoss": score, "validationSamples": val_total,
      "validationTop1": top1 / val_total, "validationTop5": top5 / val_total,
      "elapsedSeconds": time.monotonic() - started,
    }
    history.append(summary)
    reserve_space(output, 256 * 1024**2)
    if score < best_loss - .0001:
      best_loss, best_epoch, stale = score, epoch, 0
      temporary = output / "best.safetensors.tmp"
      save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, temporary)
      os.replace(temporary, output / "best.safetensors")
    else:
      stale += 1
    scheduler.step()
    # Same-run recovery only. This is not an external pretrained initialization.
    temporary = output / "last-training.pt.tmp"
    torch.save({
      "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
      "scheduler": scheduler.state_dict(), "torchRng": torch.random.get_rng_state(),
      "samplerRng": generator.get_state(), "plan": plan,
    }, temporary)
    os.replace(temporary, output / "last-training.pt")
    write_json(output / "history.json", history)
    write_json(output / "progress.json", {
      **summary, "updatedAt": now(), "stage": "epoch-complete", "optimizerSteps": steps,
      "bestEpoch": best_epoch, "classCount": len(classes),
    })
    print(json.dumps(summary), flush=True)
    if stale >= 10:
      break
  best = load_file(output / "best.safetensors", device="cpu")
  if torch.equal(best["conv1.weight"], initial_stem):
    raise ValueError("Random backbone never changed; refuse trained-model publication.")
  checkpoint = output / "classifier.safetensors"
  os.replace(output / "best.safetensors", checkpoint)
  contract = build_trained_contract(
    classes, target_species_ids=manifest["targetSpeciesIds"], training_examples=counts,
    completed_epochs=len(history), backbone_optimizer_steps=steps,
    checkpoint_sha256=sha256(checkpoint),
  )
  write_json(output / "trained-species.json", contract)
  metadata = {
    **plan, "completedAt": now(), "status": "trained-offline",
    "weightsSha256": sha256(checkpoint), "contractSha256": contract["contractSha256"],
    "classes": contract["classes"], "bestEpoch": best_epoch, "epochsRun": len(history),
    "preprocessing": {
      "imageSize": image_size, "resizeShortEdge": image_size + 24,
      "centerCrop": image_size, "mean": [.5] * 3, "std": [.5] * 3,
    },
    "validation": history[best_epoch - 1], "confidenceCalibrated": False,
    "independentFinalBenchmark": False, "productionApproved": False,
    "externalSpeciesFallbackAllowed": False, "backboneChangedFromRandomInitialization": True,
    "scope": "Species classifier only. No binary gate or full-target accuracy claim.",
  }
  write_json(output / "model.json", metadata)
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "trained-offline", "classCount": len(classes),
    "epochsRun": len(history), "bestEpoch": best_epoch, "onlineChanged": False,
  })
  print(json.dumps({"stage": "complete", "classCount": len(classes), "onlineChanged": False}))
  return metadata


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument("--image-size", type=int, default=160)
  args = parser.parse_args()
  train(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size,
        threads=args.threads, image_size=args.image_size)


if __name__ == "__main__":
  main()
