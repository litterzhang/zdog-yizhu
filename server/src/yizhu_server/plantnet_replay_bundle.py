"""Fit offline replay heads from a locally verified train/validation feature bundle."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from yizhu_server.plantnet_incremental import fit_additions
from yizhu_server.plantnet_transfer import validate_classes
from yizhu_server.training_classifier import SEED, validate_samples
from yizhu_server.training_data import now, sha256, write_json


def train_bundle(bundle, output, variant):
    import torch
    from safetensors.torch import load_file, save_file

    if output.exists() or variant not in {"observed-class", "equal-groups"}:
        raise ValueError("Use a new output directory and a registered replay variant.")
    plan_path = bundle / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (
        plan["testFeaturesIncluded"] is not False or plan["privatePhotosUsed"] is not False
        or plan["automaticDeployment"] is not False or plan["seed"] != SEED
        or plan["variants"] != ["observed-class", "equal-groups"]
        or plan["equalGroupMass"] != {"retainedA": .5, "additionalB": .5}
        or type(plan["epochs"]) is not int or not 1 <= plan["epochs"] <= 200
        or plan["versions"] != {
            key: version(key) for key in ("torch", "torchvision", "safetensors")
        }
        or sha256(bundle / "features.safetensors") != plan["featuresSha256"]
        or sha256(bundle / "samples.json") != plan["samplesSha256"]
    ):
        raise ValueError("Bundle identity, fixed settings, or environment mismatch.")
    validate_classes(plan["classes"])
    retained_count, additions = plan["retainedClassCount"], plan["additionCount"]
    if (
        type(retained_count) is not int or retained_count < 1
        or type(additions) is not int or additions < 1
        or retained_count + additions != len(plan["classes"])
    ):
        raise ValueError("Invalid retained/additional output counts.")
    provenance = json.loads((bundle / "samples.json").read_text())
    if set(provenance) != {"train", "validation"}:
        raise ValueError("Only train/validation rows may enter head optimization.")
    validate_samples(provenance["train"] + provenance["validation"], plan["classes"])
    tensors = load_file(bundle / "features.safetensors", device="cpu")
    if set(tensors) != {
        "train_features", "train_labels", "validation_features", "validation_labels",
        "retained_weight", "retained_bias",
    }:
        raise ValueError("Unexpected bundle tensors; no test features permitted.")
    class_indices = {row["speciesId"]: i for i, row in enumerate(plan["classes"])}
    splits = {}
    for split in ("train", "validation"):
        rows = provenance[split]
        x, y = tensors[split + "_features"], tensors[split + "_labels"]
        if (
            len(rows) != plan["splitCounts"][split] or not rows
            or any(row["split"] != split for row in rows)
            or x.shape != (len(rows), 2048) or x.dtype != torch.float32
            or not torch.isfinite(x).all() or not (x.norm(dim=1) > 0).all()
            or y.shape != (len(rows),) or y.dtype != torch.long
            or y.tolist() != [class_indices[row["speciesId"]] for row in rows]
        ):
            raise ValueError("Feature rows and original split/labels disagree.")
        splits[split] = (x, y)
    weight, bias = tensors["retained_weight"], tensors["retained_bias"]
    if (
        weight.shape != (retained_count, 2048) or bias.shape != (retained_count,)
        or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (weight, bias))
    ):
        raise ValueError("Invalid frozen A parameters.")
    output.mkdir(parents=True)
    trial = {
        "createdAt": now(), "bundlePlanSha256": sha256(plan_path), "variant": variant,
        "sourceSnapshotSha256": plan["sourceSnapshotSha256"],
        "baselineModelSha256": plan["baselineModelSha256"],
        "epochs": plan["epochs"], "seed": SEED, "testUsedForSelection": False,
        "automaticDeployment": False,
    }
    write_json(output / "plan.json", trial)
    print(json.dumps({"stage": "training-head", "variant": variant}), flush=True)
    head, training = fit_additions(
        splits, (weight, bias), additions, epochs=plan["epochs"],
        retained_replay=True, group_balanced_replay=variant == "equal-groups",
    )
    save_file(
        {key: value.detach().contiguous() for key, value in head.state_dict().items()},
        output / "classifier.safetensors",
    )
    with torch.inference_mode():
        ranks = head(splits["validation"][0]).topk(5, dim=1).indices
        labels = splits["validation"][1]
        validation = {
            "samples": len(labels), "top1": float(ranks[:, 0].eq(labels).float().mean()),
            "top5": float(ranks.eq(labels[:, None]).any(1).float().mean()),
        }
    write_json(output / "training.json", training)
    report = {
        **trial, "completedAt": now(), "classCount": len(plan["classes"]),
        "weightsSha256": sha256(output / "classifier.safetensors"),
        "trainingSha256": sha256(output / "training.json"),
        "validation": validation, "versions": plan["versions"],
        "scope": "Head-only offline experiment. No test scores or live configuration changes.",
    }
    write_json(output / "completed.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("observed-class", "equal-groups"), required=True)
    args = parser.parse_args()
    print(json.dumps(train_bundle(args.bundle, args.output, args.variant), indent=2))


if __name__ == "__main__":
    main()
