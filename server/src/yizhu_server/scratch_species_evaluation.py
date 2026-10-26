"""Read-only raw-image evaluation of our completed clean-start classifier."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from yizhu_server.random_init_classifier import build_random_classifier
from yizhu_server.trained_species_contract import validate_trained_contract
from yizhu_server.training_classifier import validate_samples
from yizhu_server.training_data import now, sha256, write_json


def validate_inputs(metadata, contract, training, heldout, *, training_sha, targets):
    validate_trained_contract(contract)
    classes = contract["classes"]
    identities = [(r["speciesId"], r["scientificName"]) for r in classes]
    for document in (metadata, training, heldout):
        if [(r["speciesId"], r["scientificName"]) for r in document["classes"]] != identities:
            raise ValueError("Model and data class order/identity must agree.")
    if (
        metadata["status"] != "trained-offline" or metadata["initialization"] != "random"
        or metadata["pretrainedWeightsUsed"] is not False
        or metadata["pretrainedFeaturesUsed"] is not False
        or metadata["inheritedClassCount"] != 0
        or metadata["trainBackbone"] is not True
        or metadata["backboneChangedFromRandomInitialization"] is not True
        or metadata["testUsedForSelection"] is not False
        or metadata["automaticDeployment"] is not False
        or metadata["externalSpeciesFallbackAllowed"] is not False
        or metadata["weightsSha256"] != contract["checkpointSha256"]
        or metadata["contractSha256"] != contract["contractSha256"]
        or metadata["classCount"] != len(classes)
        or metadata["manifestSha256"] != training_sha
        or heldout["trainingManifestSha256"] != training_sha
        or training["initialization"] != "random"
        or training["pretrainedWeightsAllowed"] is not False
        or training["pretrainedFeaturesAllowed"] is not False
        or training["testImagesIncluded"] is not False
        or training["targetSpeciesIds"] != contract["targetSpeciesIds"]
    ):
        raise ValueError("Require matching, completed clean-start model and frozen data.")
    rows = training["samples"] + heldout["samples"]
    validate_samples(rows, classes)
    if (
        not heldout["samples"]
        or any(r["split"] != "test" for r in heldout["samples"])
        or any(r["split"] not in {"train", "validation"} for r in training["samples"])
        or {r["speciesId"] for r in heldout["samples"]} != {sid for sid, _ in identities}
    ):
        raise ValueError("Every trained species must have held-out test images.")
    counts = Counter(r["speciesId"] for r in training["samples"] if r["split"] == "train")
    if counts != {r["speciesId"]: r["trainingExamples"] for r in classes}:
        raise ValueError("Actual training counts differ from trained support.")
    approved = {r["speciesId"]: r["scientificName"] for r in targets["classes"]}
    if len(approved) != len(targets["classes"]) or any(
        approved.get(sid) != name for sid, name in identities
    ):
        raise ValueError("Trained species fall outside the current approved target identities.")
    return classes


def summarize(predictions, classes, targets):
    """Never silently drop an unreadable image or count missing target species as tested."""
    by_id = {r["speciesId"]: r for r in classes}
    totals, top1, top5, confusions = Counter(), Counter(), Counter(), Counter()
    seen = set()
    for row in predictions:
        truth, ranked, sample = row["expectedSpeciesId"], row["rankedSpeciesIds"], row["sampleId"]
        if (
            truth not in by_id or sample in seen or len(ranked) != min(5, len(classes))
            or len(set(ranked)) != len(ranked) or not set(ranked) <= set(by_id)
        ):
            raise ValueError("Invalid, duplicate, or untrained prediction.")
        seen.add(sample)
        totals[truth] += 1
        top1[truth] += ranked[0] == truth
        top5[truth] += truth in ranked
        if ranked[0] != truth:
            confusions[(truth, ranked[0])] += 1
    if set(totals) != set(by_id):
        raise ValueError("Do not report partial-class evaluation as complete.")
    per_species = [{
        **{k: r[k] for k in ("speciesId", "scientificName", "chineseName")},
        "samples": totals[r["speciesId"]],
        "top1Correct": top1[r["speciesId"]], "top5Correct": top5[r["speciesId"]],
        "top1": top1[r["speciesId"]] / totals[r["speciesId"]],
        "top5": top5[r["speciesId"]] / totals[r["speciesId"]],
    } for r in classes]
    return {
        "sampleCount": len(predictions), "top1Correct": sum(top1.values()),
        "top5Correct": sum(top5.values()), "top1": sum(top1.values()) / len(predictions),
        "top5": sum(top5.values()) / len(predictions),
        "macroTop1": sum(r["top1"] for r in per_species) / len(classes),
        "macroTop5": sum(r["top5"] for r in per_species) / len(classes),
        "perSpecies": per_species,
        "topConfusions": [
            {"expectedSpeciesId": a, "predictedSpeciesId": b, "count": count}
            for (a, b), count in confusions.most_common(20)
        ],
        "targetCoverage": {
            "approvedSpecies": len(targets["classes"]), "evaluatedSpecies": len(classes),
            "unevaluatedSpeciesIds": [
                r["speciesId"] for r in targets["classes"] if r["speciesId"] not in by_id
            ],
            "unresolvedInputCount": targets.get("unresolvedInputCount"),
        },
        "scope": "Closed-set species-only development holdout; not full-target end-to-end.",
        "binaryGateIncluded": False, "nonPlantRejectionMeasured": False,
        "unknownSpeciesRejectionMeasured": False,
        "independentFinalBenchmark": False, "endToEndTop5": None,
        "goalAchieved": False, "productionApproved": False, "onlineChanged": False,
    }


def evaluate(model_dir, data, targets_path, output, *, threads=2, batch_size=16):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms

    if output.exists() or not 1 <= threads <= 8 or not 1 <= batch_size <= 128:
        raise ValueError("Use a fresh output directory and bounded evaluation resources.")
    metadata = json.loads((model_dir / "model.json").read_text())
    contract = json.loads((model_dir / "trained-species.json").read_text())
    training_path, test_path = data / "train-validation.json", data / "test-heldout.json"
    training, heldout = json.loads(training_path.read_text()), json.loads(test_path.read_text())
    targets = json.loads(targets_path.read_text())
    training_sha = sha256(training_path)
    classes = validate_inputs(
        metadata, contract, training, heldout, training_sha=training_sha, targets=targets,
    )
    checkpoint = model_dir / "classifier.safetensors"
    if sha256(checkpoint) != metadata["weightsSha256"]:
        raise ValueError("Trained checkpoint checksum mismatch.")
    settings = metadata["preprocessing"]
    size = settings["imageSize"]
    if (
        type(size) is not int or not 64 <= size <= 512
        or settings != {"imageSize": size, "resizeShortEdge": size + 24,
                        "centerCrop": size, "mean": [.5] * 3, "std": [.5] * 3}
    ):
        raise ValueError("Unsupported evaluation preprocessing; do not silently substitute.")
    # Audit train, validation and test files before running any test predictions.
    for row in training["samples"] + heldout["samples"]:
        path = (data / row["path"]).resolve()
        if not path.is_relative_to((data / "images").resolve()) or sha256(path) != row["imageSha256"]:
            raise ValueError("Raw image path or checksum changed.")
    torch.set_num_threads(threads)
    model = build_random_classifier(
        len(classes), architecture=metadata["architecture"], seed=metadata["seed"],
    )
    weights = load_file(checkpoint, device="cpu")
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError("Nonfinite trained parameters.")
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval()
    del weights
    transform = transforms.Compose([
        transforms.Resize(size + 24), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
    ])
    output.mkdir(parents=True)
    evidence = {
        "startedAt": now(), "modelMetadataSha256": sha256(model_dir / "model.json"),
        "weightsSha256": sha256(checkpoint), "contractSha256": contract["contractSha256"],
        "trainingManifestSha256": training_sha, "testManifestSha256": sha256(test_path),
        "targetsSha256": sha256(targets_path), "preprocessing": settings,
        "selection": metadata["selection"], "selectedEpoch": metadata["bestEpoch"],
        "validation": metadata["validation"],
        "torchVersion": torch.__version__, "device": "cpu", "threads": threads,
        "batchSize": batch_size, "rawImagesVerified": len(training["samples"]) + len(heldout["samples"]),
        "testUsedForModelSelection": False,
    }
    write_json(output / "plan.json", evidence)
    predictions, started = [], time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(heldout["samples"]), batch_size):
                rows = heldout["samples"][start:start + batch_size]
                tensors = []
                for row in rows:
                    with Image.open(data / row["path"]) as image:
                        tensors.append(transform(image.convert("RGB")))
                logits = model(torch.stack(tensors))
                if logits.shape != (len(rows), len(classes)) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid trained model output.")
                ranking = logits.argsort(dim=1, descending=True, stable=True)[:, :min(5, len(classes))]
                scores = logits.softmax(dim=1).gather(1, ranking)
                for row, indices, values in zip(rows, ranking.tolist(), scores.tolist(), strict=True):
                    predictions.append({
                        "sampleId": row["sampleId"], "imageSha256": row["imageSha256"],
                        "expectedSpeciesId": row["speciesId"],
                        "rankedSpeciesIds": [classes[i]["speciesId"] for i in indices],
                        "scores": values, "confidence": None,
                    })
                write_json(output / "progress.json", {
                    "updatedAt": now(), "stage": "evaluating", "completed": len(predictions),
                    "total": len(heldout["samples"]), "elapsedSeconds": time.monotonic() - started,
                })
        if len(predictions) != len(heldout["samples"]):
            raise ValueError("Some held-out images were not evaluated.")
        report = {**evidence, **summarize(predictions, classes, targets),
                  "completedAt": now(), "elapsedSeconds": time.monotonic() - started}
        write_json(output / "predictions.json", predictions)
        report["predictionsSha256"] = sha256(output / "predictions.json")
        write_json(output / "report.json", report)
        write_json(output / "progress.json", {
            "updatedAt": now(), "stage": "evaluated-offline",
            "samples": len(predictions), "top1": report["top1"], "top5": report["top5"],
            "goalAchieved": False, "onlineChanged": False,
        })
        return report
    except Exception as exc:
        write_json(output / "progress.json", {
            "updatedAt": now(), "stage": "failed", "errorType": type(exc).__name__,
            "completed": len(predictions), "metricsPublished": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "targets", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = evaluate(args.model, args.data, args.targets, args.output,
                      threads=args.threads, batch_size=args.batch_size)
    print(json.dumps({k: result[k] for k in (
        "sampleCount", "top1", "top5", "goalAchieved", "onlineChanged",
    )}))


if __name__ == "__main__":
    main()
