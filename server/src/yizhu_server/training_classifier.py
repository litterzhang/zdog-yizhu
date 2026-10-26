"""Frozen BioCLIP 2 + trained linear head. Offline only; never changes the API."""

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from yizhu_server.training_data import (
    DEFAULT_DATA,
    SPLIT_TARGETS,
    DuplicateIndex,
    dataset_lock,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)

SEED = 20260922


def encoder_identity():
    from yizhu_server.recognition_rerank import MODEL_ID, MODEL_REVISION, MODEL_SHA256

    return {
        "model": MODEL_ID, "revision": MODEL_REVISION, "weightsSha256": MODEL_SHA256,
        "preprocessing": "recognition_rerank.BioClipReranker.official",
        "dimension": 768, "normalized": True,
        "openClipVersion": version("open_clip_torch"), "torchVersion": version("torch"),
    }


def validate_samples(rows, species):
    """Fail closed on contamination, unlicensed samples, or conflicting labels."""
    from yizhu_server.training_data import LICENSES, group_split

    species_ids = {item["speciesId"]: item["scientificName"] for item in species}
    observers, observations, photo_ids, sample_ids = {}, set(), set(), set()
    duplicates = DuplicateIndex()
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError("Image path escapes public dataset.")
        if (
            row["speciesId"] not in species_ids
            or row["scientificName"] != species_ids[row["speciesId"]]
            or row["license"] not in LICENSES or not row.get("attribution")
            or row["split"] != group_split(row["observerGroup"])
        ):
            raise ValueError("Invalid label, license or observer split.")
        if row["observerGroup"] in observers and observers[row["observerGroup"]] != row["split"]:
            raise ValueError("Observer leaks across splits.")
        observers[row["observerGroup"]] = row["split"]
        if (
            row["observationId"] in observations or row["photoId"] in photo_ids
            or row["sampleId"] in sample_ids
        ):
            raise ValueError("Duplicate observation/photo/sample.")
        observations.add(row["observationId"])
        photo_ids.add(row["photoId"])
        sample_ids.add(row["sampleId"])
        perceptual = int(row["differenceHash"], 16)
        if duplicates.contains(row["imageSha256"], perceptual):
            raise ValueError("Exact or near-duplicate images in dataset.")
        duplicates.add(row["imageSha256"], perceptual)


def eligible_species(rows, species, split_targets=None):
    targets = SPLIT_TARGETS if split_targets is None else split_targets
    if set(targets) != set(SPLIT_TARGETS) or any(
        type(value) is not int or value < 1 for value in targets.values()
    ):
        raise ValueError("Invalid dataset split targets.")
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["speciesId"]][row["split"]] += 1
    return [
        row for row in species
        if all(counts[row["speciesId"]][key] >= minimum
               for key, minimum in targets.items())
    ]


def feature_key(row, identity):
    return hashlib.sha256(json.dumps({
        "image": row["imageSha256"], "encoder": identity,
    }, sort_keys=True).encode()).hexdigest()


def read_feature(path, dimension):
    import numpy as np

    vector = np.load(path, allow_pickle=False)
    if (
        vector.shape != (dimension,) or vector.dtype != np.float32
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1, atol=0.001)
    ):
        raise ValueError("Corrupt feature cache.")
    return vector


def extract(root, rows):
    import numpy as np
    import torch

    from yizhu_server.recognition import read_photo
    from yizhu_server.recognition_rerank import BioClipReranker

    identity = encoder_identity()
    features = root / "features"
    features.mkdir(exist_ok=True)
    encoder = None
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        photo = root / row["path"]
        if not photo.resolve().is_relative_to((root / "images").resolve()):
            raise ValueError("Image path or symlink escapes dataset.")
        if sha256(photo) != row["imageSha256"]:
            raise ValueError("Image checksum mismatch.")
        path = features / f"{feature_key(row, identity)}.npy"
        if path.exists():
            read_feature(path, identity["dimension"])
            continue
        if encoder is None:
            encoder = BioClipReranker(threads=1)
        with read_photo(photo) as image, torch.inference_mode():
            tensor = encoder.preprocess(image).unsqueeze(0)
            vector = torch.nn.functional.normalize(encoder.model.encode_image(tensor), dim=-1)
            array = vector[0].cpu().numpy().astype(np.float32)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        read_feature(temporary, identity["dimension"])
        temporary.replace(path)
        if index % 20 == 0 or index == len(rows):
            write_json(root / "progress.json", {
                "updatedAt": now(), "stage": "encoding", "completed": index,
                "total": len(rows), "elapsedSeconds": round(time.monotonic() - started, 1),
            })
            print(f"encoded {index}/{len(rows)}", flush=True)
    # Drop the large frozen model before optimizing the small head.
    del encoder
    return identity


def metrics(logits, labels, class_count):
    import torch

    logits = logits.detach()
    if not len(labels):
        raise ValueError("Cannot score an empty split.")
    prediction = logits.argmax(1)
    top_k = min(5, class_count)
    correct = prediction.eq(labels)
    per_class = []
    for index in range(class_count):
        mask = labels.eq(index)
        per_class.append(float(correct[mask].float().mean()) if mask.any() else None)
    return {
        "samples": len(labels), "top1": float(correct.float().mean()),
        f"top{top_k}": float(logits.topk(top_k, dim=1).indices.eq(labels[:, None]).any(1).float().mean()),
        "macroRecall": sum(value for value in per_class if value is not None)
        / sum(value is not None for value in per_class),
        "perClassRecall": per_class,
        "crossEntropy": float(torch.nn.functional.cross_entropy(logits, labels)),
    }


def fit_head(splits, class_count, *, epochs=200, hidden_dimension=0):
    import torch

    if type(hidden_dimension) is not int or hidden_dimension not in (0, 32):
        raise ValueError("Use a linear head or the fixed 32-unit MLP experiment.")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    head = (
        torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, class_count),
        ) if hidden_dimension else torch.nn.Linear(x_train.shape[1], class_count)
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(y_train), generator=generator)
        for batch in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = float(torch.nn.functional.cross_entropy(
                head(x_validation), y_validation,
            ))
        history.append({"epoch": epoch, "validationLoss": validation_loss})
        if validation_loss < best_loss - 0.0001:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    head.load_state_dict(best_state)
    return head, {"bestEpoch": best_epoch, "epochsRun": len(history), "history": history}


def train(root, output, *, allow_partial=False, epochs=200):
    require_dataset_active(root)
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if output.exists():
        raise ValueError("Model output already exists; choose a new version directory.")
    document = json.loads((root / "species.json").read_text())
    rows = load_samples(root)
    validate_samples(rows, document["species"])
    targets = document.get("splitTargets", SPLIT_TARGETS)
    classes = eligible_species(rows, document["species"], targets)
    if len(classes) < 2:
        raise ValueError(f"Need >=2 classes meeting dataset split targets: {targets}.")
    if len(classes) != len(document["species"]) and not allow_partial:
        raise ValueError(
            f"Only {len(classes)}/{len(document['species'])} classes ready; "
            "--allow-partial explicitly trains a labeled subset, never a fake 1,000-class model."
        )
    class_index = {row["speciesId"]: index for index, row in enumerate(classes)}
    rows = sorted(
        (row for row in rows if row["speciesId"] in class_index),
        key=lambda row: row["sampleId"],
    )
    identity = extract(root, rows)
    splits = {}
    for split in SPLIT_TARGETS:
        selected = [row for row in rows if row["split"] == split]
        features = np.stack([
            read_feature(root / "features" / f"{feature_key(row, identity)}.npy", identity["dimension"])
            for row in selected
        ])
        splits[split] = (
            torch.from_numpy(features),
            torch.tensor([class_index[row["speciesId"]] for row in selected], dtype=torch.long),
        )
    write_json(root / "progress.json", {
        "updatedAt": now(), "stage": "training", "classes": len(classes),
        "images": len(rows),
    })
    head, optimization = fit_head(splits, len(classes), epochs=epochs)
    # Centroid baseline uses training data only; final test is not used for selection.
    x_train, y_train = splits["train"]
    prototypes = torch.nn.functional.normalize(torch.stack([
        x_train[y_train == index].mean(0) for index in range(len(classes))
    ]), dim=-1)
    evaluation = {}
    with torch.inference_mode():
        for split, (features, labels) in splits.items():
            evaluation[split] = {
                "linearHead": metrics(head(features), labels, len(classes)),
                "nearestCentroid": metrics(features @ prototypes.T, labels, len(classes)),
            }
    output.mkdir(parents=True)
    weights = output / "classifier.safetensors"
    save_file({key: value.contiguous() for key, value in head.state_dict().items()}, str(weights))
    manifest = output / "training-manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "createdAt": now(), "architecture": "normalized-bioclip2-linear-v1", "encoder": identity,
        "dataPolicy": document.get("policy"),
        "trainingScope": document.get("purpose", "Bounded species classification experiment"),
        "dataLimitations": document.get("limitations", []),
        "catalogSha256": document.get("catalogSha256"),
        "seed": SEED, "classCount": len(classes), "requestedClassCount": len(document["species"]),
        "splitTargets": targets,
        "classes": classes, "weightsSha256": sha256(weights),
        "shortlistSha256": sha256(root / "species.json"), "manifestSha256": sha256(manifest),
        "optimizer": {"name": "AdamW", "lr": 0.01, "weightDecay": 0.01, "batchSize": 256},
        "evaluation": evaluation, "optimization": optimization,
        "status": "experimental-offline-not-deployed", "calibrated": False,
        "limitations": [
            "Closed set only; no nonplant/out-of-distribution rejection trained.",
            "Public research-grade labels not expert audited; not private-user accuracy.",
            "Observer-grouped and deduplicated, but individual identity is not verified.",
            "Overlap with foundation-model pretraining is not ruled out.",
            "Centroid comparison is not a comparison against the complete online cascade.",
            "No production publication or online model replacement.",
        ],
    }
    write_json(output / "model.json", metadata)
    from yizhu_server.solution_metadata import write_artifact_metadata

    write_artifact_metadata(output / "model.json")
    write_json(root / "progress.json", {
        "updatedAt": now(), "stage": "trained-offline", "classes": len(classes),
        "requestedClasses": len(document["species"]), "output": str(output),
        "test": evaluation["test"],
    })
    print(json.dumps({
        "classes": len(classes), "output": str(output), "test": evaluation["test"],
    }, ensure_ascii=False), flush=True)
    return metadata


def predict(model_path, photo):
    import torch
    from safetensors.torch import load_file

    from yizhu_server.recognition import read_photo
    from yizhu_server.recognition_rerank import BioClipReranker

    metadata = json.loads((model_path / "model.json").read_text())
    if (
        metadata["encoder"] != encoder_identity()
        or sha256(model_path / "classifier.safetensors") != metadata["weightsSha256"]
    ):
        raise ValueError("Model/encoder version or checksum mismatch.")
    head = torch.nn.Linear(metadata["encoder"]["dimension"], metadata["classCount"])
    head.load_state_dict(load_file(str(model_path / "classifier.safetensors")), strict=True)
    head.eval()
    encoder = BioClipReranker(threads=1)
    with read_photo(photo) as image, torch.inference_mode():
        features = torch.nn.functional.normalize(
            encoder.model.encode_image(encoder.preprocess(image).unsqueeze(0)), dim=-1,
        )
        logits, indices = head(features)[0].topk(min(5, metadata["classCount"]))
    print(json.dumps({
        "status": "requires_review", "scope": "experimental-closed-set",
        "candidates": [{
            **metadata["classes"][index], "logit": float(score), "confidence": None,
        } for score, index in zip(logits, indices.tolist(), strict=True)],
    }, ensure_ascii=False, indent=2))


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True, help="Versioned model directory")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--photo", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000:
        parser.error("--epochs must be 1–1000.")
    if args.action == "train":
        with dataset_lock(args.data):
            train(args.data, args.output, allow_partial=args.allow_partial, epochs=args.epochs)
    elif args.photo is None:
        parser.error("predict requires --photo.")
    else:
        predict(args.output, args.photo)


if __name__ == "__main__":
    main()
