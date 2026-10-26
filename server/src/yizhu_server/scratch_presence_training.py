"""Train a whole-image plant presence gate from random weights, without species labels."""

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from yizhu_server.plant_presence import LABELS, binary_metrics, load_dataset, select_threshold
from yizhu_server.random_init_classifier import build_random_classifier
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import now, require_dataset_active, sha256, write_json


def full_frame(image, size):
    from PIL import Image, ImageOps

    resized = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def validate_plan(document):
    if (
        not isinstance(document, dict)
        or document.get("task") != "plant-presence"
        or document.get("initialization") != "random"
        or document.get("labels") != LABELS
        or any(document.get(key) is not False for key in (
            "pretrainedWeightsAllowed", "pretrainedFeaturesAllowed",
            "testImagesIncluded", "automaticDeployment",
        ))
        or type(document.get("seed")) is not int or not 0 <= document["seed"] < 2**63
        or not isinstance(document.get("samples"), list) or not document["samples"]
    ):
        raise ValueError("Require a clean-start binary-only dataset without test images.")
    for row in document["samples"]:
        if (
            not isinstance(row, dict) or row.get("split") not in {"train", "validation"}
            or row.get("label") not in LABELS or not row.get("labelEvidence")
        ):
            raise ValueError("Only explicit human plant/nonplant labels in train/validation.")


def train(data, output, *, epochs=30, threads=1, batch_size=16, image_size=160,
          minimum_recall=.95):
    require_dataset_active(data)

    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    if (
        output.exists() or not 1 <= epochs <= 100 or not 1 <= threads <= 4
        or not 2 <= batch_size <= 128 or not 64 <= image_size <= 512
        or not 0 < minimum_recall <= 1
    ):
        raise ValueError("Use new output and bounded training settings.")
    manifest = json.loads((data / "samples.json").read_text())
    validate_plan(manifest)
    rows, manifest_sha = load_dataset(data, required_splits=("train", "validation"))
    seed = manifest["seed"]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = build_random_classifier(2, architecture="resnet18", seed=seed)
    initial_stem = model.conv1.weight.detach().clone()
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.02),
        transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
    ])
    deterministic = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
    ])

    class Images(Dataset):
        def __init__(self, split):
            self.rows = [r for r in rows if r["split"] == split]
            self.transform = augment if split == "train" else deterministic

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            with Image.open(data / row["path"]) as image:
                tensor = self.transform(full_frame(image, image_size))
            return tensor, LABELS.index(row["label"])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Images("train"), batch_size=batch_size, shuffle=True, num_workers=0, generator=generator,
    )
    val_loader = DataLoader(Images("validation"), batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    output.mkdir(parents=True)
    reserve_space(output, 64 * 1024**2)
    plan = {
        "createdAt": now(), "task": "plant-presence", "architecture": "resnet18",
        "initialization": "random", "pretrainedWeightsUsed": False, "pretrainedFeaturesUsed": False,
        "labels": LABELS, "seed": seed, "manifestSha256": manifest_sha,
        "trainImages": len(train_loader.dataset), "validationImages": len(val_loader.dataset),
        "counts": {split: dict(Counter(r["label"] for r in rows if r["split"] == split))
                   for split in ("train", "validation")},
        "trainBackbone": True, "epochs": epochs, "batchSize": batch_size, "threads": threads,
        "selection": "lowest validation cross entropy; patience 10",
        "minimumValidationPlantRecall": minimum_recall,
        "testUsedForSelection": False, "automaticDeployment": False,
        "preprocessing": {"mode": "whole-image-letterbox", "imageSize": image_size,
                          "paddingRgb": [128, 128, 128], "mean": [.5] * 3, "std": [.5] * 3},
    }
    write_json(output / "plan.json", plan)
    history, best_loss, best_epoch, stale, steps = [], float("inf"), 0, 0, 0
    started = time.monotonic()

    def validate():
        model.eval()
        scores, labels, loss_sum = [], [], 0.
        with torch.inference_mode():
            for images, truth in val_loader:
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite validation output.")
                loss_sum += float(torch.nn.functional.cross_entropy(logits, truth, reduction="sum"))
                scores.extend(logits.softmax(1)[:, 1].tolist())
                labels.extend(truth.tolist())
        return loss_sum / len(labels), scores, labels

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, consumed = 0., 0
        for batch, (images, truth) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), truth)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if model.conv1.weight.grad is None:
                raise ValueError("Binary backbone not participating in training.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(truth)
            consumed += len(truth)
            steps += 1
            if batch == 1 or batch % 10 == 0:
                write_json(output / "progress.json", {
                    "updatedAt": now(), "stage": "training", "epoch": epoch,
                    "maxEpochs": epochs, "batch": batch, "batches": len(train_loader),
                    "optimizerSteps": steps, "trainLoss": total_loss / consumed,
                    "elapsedSeconds": time.monotonic() - started,
                })
        if consumed != len(train_loader.dataset):
            raise ValueError("Training failed to consume the full frozen dataset.")
        val_loss, scores, labels = validate()
        if not math.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss.")
        summary = {
            "epoch": epoch, "trainLoss": total_loss / consumed, "validationLoss": val_loss,
            "validationAtHalf": binary_metrics(scores, labels, .5),
            "elapsedSeconds": time.monotonic() - started,
        }
        if val_loss < best_loss - .0001:
            reserve_space(output, 64 * 1024**2)
            best_loss, best_epoch, stale = val_loss, epoch, 0
            save_file(model.state_dict(), output / "best.safetensors.tmp")
            os.replace(output / "best.safetensors.tmp", output / "best.safetensors")
        else:
            stale += 1
        scheduler.step()
        history.append(summary)
        write_json(output / "history.json", history)
        print(json.dumps(summary), flush=True)
        if stale >= 10:
            break
    weights = load_file(output / "best.safetensors", device="cpu")
    if torch.equal(weights["conv1.weight"], initial_stem):
        raise ValueError("Backbone never changed from random initialization.")
    model.load_state_dict(weights, strict=True)
    _, scores, labels = validate()
    threshold = select_threshold(scores, labels, minimum_recall)
    checkpoint = output / "classifier.safetensors"
    os.replace(output / "best.safetensors", checkpoint)
    metadata = {
        **plan, "completedAt": now(), "status": "trained-offline",
        "epochsRun": len(history), "bestEpoch": best_epoch, "backboneOptimizerSteps": steps,
        "weightsSha256": sha256(checkpoint), "threshold": threshold,
        "thresholdSelectedOn": "validation-only",
        "validationAtSelectedThreshold": binary_metrics(scores, labels, threshold),
        "confidenceCalibrated": False, "productionApproved": False,
        "independentFinalBenchmark": False, "speciesRecognitionIncluded": False,
        "endToEndTop5": None, "goalAchieved": False,
    }
    write_json(output / "model.json", metadata)
    write_json(output / "progress.json", {
        "updatedAt": now(), "stage": "trained-offline", "epochsRun": len(history),
        "bestEpoch": best_epoch, "onlineChanged": False,
    })
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--minimum-recall", type=float, default=.95)
    args = parser.parse_args()
    train(args.data, args.output, epochs=args.epochs, threads=args.threads,
          batch_size=args.batch_size, image_size=args.image_size, minimum_recall=args.minimum_recall)


if __name__ == "__main__":
    main()
