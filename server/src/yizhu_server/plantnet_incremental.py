"""Retain pretrained A logits and learn B rows; no automatic support publication."""

import argparse
import gc
import json
import tempfile
from pathlib import Path

from yizhu_server.plantnet_pretrained import DEFAULT_DIRECTORY, verify
from yizhu_server.plantnet_training_data import DEFAULT_SETS, union_targets
from yizhu_server.plantnet_transfer import verified_photo
from yizhu_server.snapshot_files import copy_snapshot_file
from yizhu_server.training_classifier import SEED, eligible_species, validate_samples
from yizhu_server.training_data import (
    SPLIT_TARGETS,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)


def retained_parameters_for(encoder, classes):
    """Original classifier indices, IDs, names and parameters must all agree."""
    indices = []
    for row in classes:
        refs = row.get("sourceClasses", [])
        if len(refs) != 1:
            raise ValueError("Retained class needs exactly one unambiguous source output.")
        ref = refs[0]
        index = ref["classIndex"]
        if type(index) is not int or not 0 <= index < len(encoder.original_classes):
            raise ValueError("Invalid retained source class index.")
        original = encoder.original_classes[index]
        if any(original[key] != ref[key] for key in ("sourceSpeciesId", "sourceScientificName")):
            raise ValueError("Retained label differs from its pretrained source.")
        indices.append(index)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Duplicate or empty retained output set.")
    return (
        encoder.original_head.weight[indices].detach().clone(),
        encoder.original_head.bias[indices].detach().clone(),
    ), indices


def verify_retained_parameters(encoder, head, metadata):
    import torch

    count = metadata.get("retainedClassCount")
    added = metadata.get("trainedAdditionalClassCount")
    if (
        type(count) is not int or type(added) is not int or count < 1 or added < 1
        or count + added != metadata["classCount"]
    ):
        raise ValueError("Invalid retained/additional class counts.")
    (weight, bias), indices = retained_parameters_for(encoder, metadata["classes"][:count])
    if (
        metadata.get("retainedSourceIndices") != indices
        or not torch.equal(head.weight[:count].detach(), weight)
        or not torch.equal(head.bias[:count].detach(), bias)
    ):
        raise ValueError("Retained pretrained parameters changed.")


def replay_class_weights(labels, retained_count, addition_count, *, group_balanced=False):
    """Macro CE by observed class, optionally with fixed equal A/B group mass."""
    import torch

    total = retained_count + addition_count
    if (
        labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
        or retained_count < 1 or addition_count < 1
        or int(labels.min()) < 0 or int(labels.max()) >= total
    ):
        raise ValueError("Invalid replay labels or class counts.")
    frequencies = torch.bincount(labels, minlength=total)
    observed = frequencies > 0
    if not observed[:retained_count].any() or not observed[retained_count:].all():
        raise ValueError("Replay requires observed A and every trained B class.")
    weights = torch.zeros(total)
    if group_balanced:
        for start, end in ((0, retained_count), (retained_count, total)):
            indices = torch.arange(start, end)[observed[start:end]]
            weights[indices] = len(labels) / (
                2 * len(indices) * frequencies[indices].float()
            )
    else:
        weights[observed] = len(labels) / (
            int(observed.sum()) * frequencies[observed].float()
        )
    return weights


def fit_additions(splits, retained, addition_count, *, epochs=200, retained_replay=False,
                  group_balanced_replay=False):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    weight, bias = retained
    weight, bias = weight.detach().clone(), bias.detach().clone()
    count, dimension = weight.shape
    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    if (
        addition_count < 1 or not 1 <= epochs <= 1000 or bias.shape != (count,)
        or x_train.shape[1] != dimension or x_val.shape[1] != dimension
        or not len(y_train) or not len(y_val)
        or set(y_train[y_train >= count].tolist()) != set(range(count, count + addition_count))
        or int(y_train.min()) < 0
        or int(y_val.min()) < 0 or int(y_val.max()) >= count + addition_count
    ):
        raise ValueError("Require every B training label and valid A/B labels and dimensions.")
    replay_labels = y_train[y_train < count]
    if not retained_replay and len(replay_labels):
        raise ValueError("Train only B unless retained replay is explicitly enabled.")
    if retained_replay and not len(replay_labels):
        raise ValueError("Retained replay requires A train-split photos, not validation/test photos.")
    if group_balanced_replay and not retained_replay:
        raise ValueError("Group-balanced replay requires retained replay.")
    # Balance observed classes without changing labels or optimizing any A parameter.
    # For a uniform sample permutation, the mean weighted loss estimates macro CE.
    class_weights = None
    if retained_replay:
        class_weights = replay_class_weights(
            y_train, count, addition_count, group_balanced=group_balanced_replay,
        )
    new = torch.nn.Linear(dimension, addition_count)
    optimizer = torch.optim.AdamW(new.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(SEED)
    val_groups = [y_val == label for label in y_val.unique()]
    best_loss, stale, best_state, best_epoch = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        new.train()
        for batch in torch.randperm(len(y_train), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            logits = torch.cat((F.linear(x_train[batch], weight, bias), new(x_train[batch])), dim=1)
            loss = (
                F.cross_entropy(logits, y_train[batch], weight=class_weights,
                                reduction="none").mean()
                if retained_replay else F.cross_entropy(logits, y_train[batch])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new.parameters(), 5.)
            optimizer.step()
        new.eval()
        with torch.inference_mode():
            logits = torch.cat((F.linear(x_val, weight, bias), new(x_val)), dim=1)
            losses = F.cross_entropy(logits, y_val, reduction="none")
            val_loss = float(torch.stack([losses[mask].mean() for mask in val_groups]).mean())
        history.append({"epoch": epoch, "validationMacroLoss": val_loss})
        if val_loss < best_loss - .0001:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in new.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    new.load_state_dict(best_state)
    head = torch.nn.Linear(dimension, count + addition_count)
    with torch.no_grad():
        head.weight.copy_(torch.cat((weight, new.weight), dim=0))
        head.bias.copy_(torch.cat((bias, new.bias), dim=0))
    if not torch.equal(head.weight[:count], weight) or not torch.equal(head.bias[:count], bias):
        raise ValueError("Retained parameters changed during export.")
    head.eval()
    return head, {
        "bestEpoch": best_epoch, "epochsRun": len(history), "history": history,
        "optimizer": "AdamW(new rows only)", "learningRate": .001, "weightDecay": .0001,
        "validationSelection": "macro cross entropy over observed A/B validation species",
        "retainedValidationSpecies": int(y_val[y_val < count].unique().numel()),
        "retainedParametersUnchanged": True, "testUsedForSelection": False,
        "retainedReplay": {
            "enabled": retained_replay, "trainingPhotos": len(replay_labels),
            "trainingSpecies": int(replay_labels.unique().numel()),
            "retainedClassCount": count, "split": "train",
            "loss": (
                "equal-group-class-balanced-cross-entropy" if group_balanced_replay
                else "observed-class-balanced-cross-entropy" if retained_replay else "cross-entropy"
            ),
            "groupBalanced": group_balanced_replay,
            "trainingLossMassA": (
                float(class_weights[y_train[y_train < count]].sum()) / len(y_train)
                if retained_replay else 0.0
            ),
            "allRetainedSpeciesCovered": int(replay_labels.unique().numel()) == count,
        },
    }


def freeze_incremental(source, destination, retained_source, *, sets_path=DEFAULT_SETS,
                       retained_replay=False, reflink=False):
    require_dataset_active(source)
    if destination.exists():
        raise ValueError("Snapshot exists; use a new version.")
    sets = json.loads(sets_path.read_text())
    manifest = verify(DEFAULT_DIRECTORY, "resnet50")
    if sets["sourceFiles"] != manifest["files"]:
        raise ValueError("Support sets and pretrained weights do not agree.")
    union = union_targets(sets)
    a = [row for row in union if row["classRole"] == "retained-pretrained"]
    b = [row for row in union if row["classRole"] == "finetune-addition"]
    a.sort(key=lambda row: row["sourceClasses"][0]["classIndex"])
    a_ids, b_ids = {r["speciesId"] for r in a}, {r["speciesId"] for r in b}
    document = json.loads((source / "species.json").read_text())
    if (
        {r["speciesId"] for r in document["species"]} != b_ids
        or document.get("plantSetsSha256") != sha256(sets_path)
    ):
        raise ValueError("B dataset and frozen support/addition plan differ.")
    targets = document.get("splitTargets", SPLIT_TARGETS)
    sources = [(source, b_ids), (retained_source, a_ids)]
    rows, locations, source_hashes = [], {}, []
    for folder, allowed in sources:
        source_hashes.append({"folder": str(folder), "speciesSha256": sha256(folder / "species.json")})
        for row in load_samples(folder):
            if row["speciesId"] not in allowed:
                continue
            if row["sampleId"] in locations:
                raise ValueError("Duplicate sample across A/B snapshots.")
            verified_photo(folder, row)
            rows.append(row)
            locations[row["sampleId"]] = folder
    rows.sort(key=lambda row: row["sampleId"])
    validate_samples(rows, a + b)
    ready = eligible_species(rows, b, targets)
    if len(ready) < 2:
        raise ValueError("Need at least two ready additional species.")
    if not any(row["speciesId"] in a_ids and row["split"] == "validation" for row in rows):
        raise ValueError("Need some A validation photos to monitor ranking interference.")
    if retained_replay and not any(
        row["speciesId"] in a_ids and row["split"] == "train" for row in rows
    ):
        raise ValueError("Retained replay requires A train-split photos.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incremental-snapshot-", dir=destination.parent) as work:
        stage = Path(work) / "data"
        stage.mkdir()
        for row in rows:
            target = stage / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_snapshot_file(
                verified_photo(locations[row["sampleId"]], row), target, reflink=reflink,
            )
            verified_photo(stage, row)
        for item in source_hashes:
            if sha256(Path(item["folder"]) / "species.json") != item["speciesSha256"]:
                raise ValueError("Source target definition changed during snapshot.")
        write_json(stage / "species.json", {
            **document,
            "policy": (
                "plantnet-retained-A-replay-new-B-snapshot-v1" if retained_replay
                else "plantnet-retained-A-new-B-snapshot-v1"
            ),
            "species": a + b, "retainedReplayAllowed": retained_replay,
            "requestedClassCount": len(a) + len(b), "retainedClasses": a,
            "purpose": "Retain pretrained A parameters and train additional B output rows",
            "limitations": [
                *document.get("limitations", []),
                (
                    "A train photos constrain B outputs; no duplicate A labels or A parameter updates."
                    if retained_replay else
                    "A photos are for validation/test only, not training new copies of existing classes."
                ),
                "A parameters unchanged does not guarantee unchanged Top-5 with B competitors.",
                "Public development snapshot, not final independent China field benchmark.",
            ],
        })
        write_json(stage / "samples/frozen.json", {"samples": rows})
        write_json(stage / "snapshot.json", {
            "createdAt": now(), "format": 1, "speciesSha256": sha256(stage / "species.json"),
            "samplesSha256": sha256(stage / "samples/frozen.json"), "sources": source_hashes,
            "requestedClassCount": len(a) + len(b), "retainedClassCount": len(a),
            "readyAdditionalClassCount": len(ready), "imageCount": len(rows),
            "photoCopyMode": "reflink" if reflink else "copy",
            "independentFinalBenchmark": False,
        })
        stage.rename(destination)
    gc.collect()
    return json.loads((destination / "snapshot.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retained-data", type=Path, required=True)
    parser.add_argument("--sets", type=Path, default=DEFAULT_SETS)
    parser.add_argument("--retained-replay", action="store_true",
                        help="Allow A train photos as replay while keeping A parameters frozen")
    parser.add_argument("--reflink", action="store_true",
                        help="Require independent CoW photo copies; never hard-link or silently copy")
    args = parser.parse_args()
    print(json.dumps(freeze_incremental(
        args.source, args.data, args.retained_data, sets_path=args.sets,
        retained_replay=args.retained_replay, reflink=args.reflink,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
