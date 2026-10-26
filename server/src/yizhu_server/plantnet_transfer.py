"""Offline Pl@ntNet transfer classifier and conservative binary gate evaluation."""

import argparse
import gc
import hashlib
import json
import math
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

from yizhu_server.plant_presence import (
    DIMENSION,
    LABELS,
    PlantPresenceClassifier,
    binary_metrics,
    cached_feature,
    load_dataset,
    select_threshold,
)
from yizhu_server.plantnet_pretrained import (
    DEFAULT_DIRECTORY,
    PREPROCESS,
    PlantNetPretrained,
    write_report,
)
from yizhu_server.recognition import read_photo
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_classifier import (
    SEED,
    eligible_species,
    fit_head,
    validate_samples,
)
from yizhu_server.training_data import (
    ROOT,
    SPLIT_TARGETS,
    DuplicateIndex,
    dataset_lock,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)

ARCHITECTURE = "normalized-plantnet-resnet50-linear-v1"
INCREMENTAL_ARCHITECTURE = "plantnet-resnet50-retained-plus-new-v1"
DEFAULT_PRESENCE = (
    ROOT / ".cache/models/plant-presence/comparison-v2/imagenet-resnet50-linear.json"
)
DEFAULT_BINARY_DATA = ROOT / ".cache/training/plant-presence-openimages-v2"
MINIMUM_RECALL = 0.99


def verified_photo(root, row):
    relative = Path(row["path"])
    path = (root / relative).resolve()
    if (
        relative.is_absolute() or ".." in relative.parts
        or not path.is_relative_to((root / "images").resolve())
        or sha256(path) != row["imageSha256"]
    ):
        raise ValueError("Public photo path or checksum differs from the frozen manifest.")
    return path


def freeze_dataset(source, destination):
    """Read atomic per-class files without stopping the independent live collector."""
    require_dataset_active(source)
    if destination.exists():
        raise ValueError("Snapshot exists; never overwrite a frozen experiment.")
    species_bytes = (source / "species.json").read_bytes()
    document = json.loads(species_bytes)
    rows = sorted(load_samples(source), key=lambda row: row["sampleId"])
    validate_samples(rows, document["species"])
    targets = document.get("splitTargets", SPLIT_TARGETS)
    classes = eligible_species(rows, document["species"], targets)
    if len(classes) < 6:
        raise ValueError("Need at least six ready species for a meaningful Top-5 trial.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plantnet-snapshot-", dir=destination.parent) as work:
        staging = Path(work) / "data"
        (staging / "images").mkdir(parents=True)
        for row in rows:
            photo = verified_photo(source, row)
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(staging).free < 4 * 1024**3:
                raise OSError("Keep at least 4 GiB free while freezing public images.")
            shutil.copyfile(photo, target)
            verified_photo(staging, row)
        if (source / "species.json").read_bytes() != species_bytes:
            raise ValueError("Source species selection changed during snapshot.")
        (staging / "species.json").write_bytes(species_bytes)
        write_json(staging / "samples/frozen.json", {"samples": rows})
        write_json(staging / "snapshot.json", {
            "createdAt": now(), "format": 1,
            "speciesSha256": sha256(staging / "species.json"),
            "samplesSha256": sha256(staging / "samples/frozen.json"),
            "requestedClassCount": len(document["species"]),
            "readyClassCount": len(classes), "imageCount": len(rows),
            "scope": "Current licensed-photo collection snapshot, not final China field benchmark",
            "independentFinalBenchmark": False,
            "sourcePolicy": document.get("policy"),
        })
        staging.rename(destination)
    return json.loads((destination / "snapshot.json").read_text())


def frozen_data(root):
    require_dataset_active(root)
    snapshot = json.loads((root / "snapshot.json").read_text())
    if (
        snapshot.get("format") != 1
        or sha256(root / "species.json") != snapshot["speciesSha256"]
        or sha256(root / "samples/frozen.json") != snapshot["samplesSha256"]
        or set(p.name for p in (root / "samples").glob("*.json")) != {"frozen.json"}
    ):
        raise ValueError("Frozen dataset manifest changed.")
    document = json.loads((root / "species.json").read_text())
    rows = load_samples(root)
    validate_samples(rows, document["species"])
    for row in rows:
        verified_photo(root, row)
    return document, rows, snapshot


class SpeciesEncoder:
    """Official original-1081-class backbone, discarded FC, frozen normalized features."""

    def __init__(self, assets=DEFAULT_DIRECTORY, threads=1, normalize=True):
        import torch

        pretrained = PlantNetPretrained(assets, architecture="resnet50", threads=threads)
        self.model, self.preprocess = pretrained.model, pretrained.preprocess
        self.original_head = self.model.fc
        self.original_classes = pretrained.records
        self.normalize = normalize
        self.model.fc = torch.nn.Identity()
        self.identity = {
            "architecture": "resnet50", "dimension": DIMENSION,
            "weightsSha256": pretrained.manifest["files"]["resnet50_weights_best_acc.tar"]["sha256"],
            "sourceVariant": "official-1081-class-pretrained-not-V2",
            "preprocessing": {**PREPROCESS, "normalizedFeatures": normalize},
        }

    def encode(self, image):
        import torch

        with torch.inference_mode(), torch.backends.nnpack.flags(enabled=False):
            vector = self.model(self.preprocess(image).unsqueeze(0))
            if (
                vector.shape != (1, DIMENSION) or not torch.isfinite(vector).all()
                or float(vector.norm()) <= 0
            ):
                raise ValueError("Invalid species encoder feature.")
            return torch.nn.functional.normalize(vector, dim=1)[0] if self.normalize else vector[0]


def feature(encoder, root, row):
    preprocess = encoder.identity.get("preprocessing")
    if isinstance(preprocess, dict) and preprocess.get("normalizedFeatures") is False:
        import numpy as np
        import torch

        identity = {**encoder.identity, "torch": version("torch"), "torchvision": version("torchvision")}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = root / "features" / key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{row['imageSha256']}.npy"
        if not path.exists():
            reserve_space(directory, DIMENSION * 4 + 1024)
            with read_photo(root / row["path"]) as image:
                vector = encoder.encode(image).cpu().numpy().astype(np.float32)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vector, allow_pickle=False)
            temporary.replace(path)
        if not 0 < path.stat().st_size < DIMENSION * 4 + 1024:
            raise ValueError("Invalid raw feature cache size.")
        vector = np.load(path, allow_pickle=False)
        if (
            vector.shape != (DIMENSION,) or vector.dtype != np.float32
            or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0
        ):
            raise ValueError("Invalid raw feature cache.")
        return torch.from_numpy(vector)
    return cached_feature(encoder, root, {**row, "sha256": row["imageSha256"]})


def species_vectors(encoder, root, rows, classes, progress):
    import torch

    indices = {item["speciesId"]: i for i, item in enumerate(classes)}
    vectors = []
    for index, row in enumerate(rows, 1):
        require_dataset_active(root)
        vectors.append(feature(encoder, root, row))
        if index % 25 == 0 or index == len(rows):
            progress({"stage": "encoding-species", "completed": index, "total": len(rows)})
    return torch.stack(vectors), torch.tensor([indices[row["speciesId"]] for row in rows])


def conservative_threshold(binary_scores, binary_labels, species_scores,
                           original_threshold, minimum_recall=MINIMUM_RECALL):
    """Freeze from validation only; never make the existing gate stricter."""
    if (
        not species_scores or not 0 < minimum_recall <= 1
        or not math.isfinite(original_threshold) or not 0 <= original_threshold <= 1
        or any(not math.isfinite(s) or not 0 <= s <= 1 for s in species_scores)
    ):
        raise ValueError("Invalid gate calibration scores or recall target.")
    binary_threshold = select_threshold(binary_scores, binary_labels, minimum_recall)
    ordered = sorted(species_scores, reverse=True)
    species_threshold = ordered[math.ceil(len(ordered) * minimum_recall) - 1]
    return min(original_threshold, binary_threshold, species_threshold)


def calibrate_gate(classifier, binary_root, root, validation_rows, progress):
    """Only validation scores enter calibration. Previously published test scores are not read."""
    binary_rows, digest = load_dataset(binary_root)
    if digest != classifier.report["datasetSha256"]:
        raise ValueError("Binary calibration dataset differs from the frozen head dataset.")
    binary_cases = []
    for row in binary_rows:
        if row["split"] != "validation":
            continue
        result = classifier.predict_vector(cached_feature(classifier.encoder, binary_root, row))
        binary_cases.append({"id": row["id"], "label": row["label"], "score": result["score"]})
        if len(binary_cases) % 100 == 0:
            progress({"stage": "calibrating-gate", "binaryValidationImages": len(binary_cases)})
    species_cases = []
    for row in validation_rows:
        result = classifier.predict_vector(feature(classifier.encoder, root, row))
        species_cases.append({"sampleId": row["sampleId"], "score": result["score"]})
    binary_scores = [row["score"] for row in binary_cases]
    binary_labels = [LABELS.index(row["label"]) for row in binary_cases]
    species_scores = [row["score"] for row in species_cases]
    threshold = conservative_threshold(
        binary_scores, binary_labels, species_scores, classifier.report["threshold"],
    )
    return {
        "headSha256": classifier.version, "encoder": classifier.report["encoder"],
        "threshold": threshold, "originalThreshold": classifier.report["threshold"],
        "minimumValidationPlantRecall": MINIMUM_RECALL,
        "policy": "min(original, binary-validation-99%-recall, species-validation-99%-recall)",
        "binaryDatasetSha256": digest, "testUsedForSelection": False,
        "binaryValidation": binary_metrics(binary_scores, binary_labels, threshold),
        "speciesValidation": {
            "samples": len(species_scores),
            "plantRecall": sum(s >= threshold for s in species_scores) / len(species_scores),
        },
        "binaryValidationCases": binary_cases, "speciesValidationCases": species_cases,
        "confidenceCalibrated": False,
    }


def check_cross_dataset_leakage(species_rows, binary_rows):
    """Refuse cross-split duplicate photos rather than erase inconvenient test cases."""
    indices = {split: DuplicateIndex() for split in ("train", "validation", "test")}
    for row in binary_rows:
        indices[row["split"]].add(row["sha256"], int(row["differenceHash"], 16))
    for row in species_rows:
        for split, index in indices.items():
            if split != row["split"] and index.contains(
                row["imageSha256"], int(row["differenceHash"], 16),
            ):
                raise ValueError("Species/binary datasets contain cross-split duplicate photos.")


def validate_classes(classes):
    if (
        len(classes) < 6
        or any(not isinstance(row.get("speciesId"), str) or not row["speciesId"]
               or not isinstance(row.get("scientificName"), str) or not row["scientificName"]
               for row in classes)
        or len({row["speciesId"] for row in classes}) != len(classes)
        or len({row["scientificName"] for row in classes}) != len(classes)
    ):
        raise ValueError("Need >=6 distinct stable species IDs and scientific names.")


def train(root, output, *, presence_head=DEFAULT_PRESENCE, binary_root=DEFAULT_BINARY_DATA,
          allow_partial=False, epochs=200, threads=1, incremental=False, retained_replay=False,
          group_balanced_replay=False):
    import torch
    from safetensors.torch import save_file

    if output.exists() or type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("Use a new model directory and 1–1000 epochs.")
    if group_balanced_replay and (not incremental or not retained_replay):
        raise ValueError("Group-balanced replay requires incremental retained replay.")
    document, rows, snapshot = frozen_data(root)
    if retained_replay and (
        not incremental or not document.get("retainedReplayAllowed", False)
    ):
        raise ValueError("Retained replay requires incremental mode and an explicitly allowed snapshot.")
    binary_rows, binary_digest = load_dataset(binary_root)
    check_cross_dataset_leakage(rows, binary_rows)
    targets = document.get("splitTargets", SPLIT_TARGETS)
    retained = document.get("retainedClasses", []) if incremental else []
    if incremental and not retained:
        raise ValueError("Incremental training requires a retained-A snapshot.")
    retained_ids = {row["speciesId"] for row in retained}
    if retained_replay and not any(
        row["split"] == "train" and row["speciesId"] in retained_ids for row in rows
    ):
        raise ValueError("Retained replay requires A train-split photos.")
    additions = eligible_species(
        rows, [row for row in document["species"] if row["speciesId"] not in retained_ids], targets,
    )
    classes = retained + additions
    if incremental and len(additions) < 2:
        raise ValueError("Need at least two ready new species; never relabel a plan as trained.")
    validate_classes(classes)
    if len(classes) != len(document["species"]) and not allow_partial:
        raise ValueError("Not all target species ready; explicitly pass --allow-partial.")
    output.mkdir(parents=True)

    def progress(details):
        write_json(output / "progress.json", {**details, "updatedAt": now()})
        print(json.dumps(details, ensure_ascii=False), flush=True)

    # Freeze the trial settings before any model scoring.
    write_report(output / "plan.json", {
        "createdAt": now(), "snapshotSha256": sha256(root / "snapshot.json"),
        "speciesSha256": snapshot["speciesSha256"], "samplesSha256": snapshot["samplesSha256"],
        "classIds": [row["speciesId"] for row in classes],
        "requestedClassCount": len(document["species"]), "splitTargets": targets,
        "presenceHeadSha256": sha256(presence_head), "epochs": epochs, "seed": SEED,
        "binaryDatasetSha256": binary_digest,
        "minimumValidationPlantRecall": MINIMUM_RECALL, "trainingMode": "frozen-backbone",
        "retainedClassCount": len(retained), "incremental": incremental,
        "retainedReplay": retained_replay,
        "groupBalancedReplay": group_balanced_replay,
        "acceptance": "End-to-end Top-5 >=90%, full target coverage, independent field benchmark",
    })
    encoder = SpeciesEncoder(threads=threads, normalize=not incremental)
    identity = encoder.identity
    retained_parameters, source_indices = None, []
    if incremental:
        from yizhu_server.plantnet_incremental import retained_parameters_for

        retained_parameters, source_indices = retained_parameters_for(encoder, retained)
    class_ids = {row["speciesId"] for row in classes}
    splits = {}
    for split in ("train", "validation"):
        selected = [
            row for row in rows if row["split"] == split and row["speciesId"] in class_ids
            and (split != "train" or retained_replay or row["speciesId"] not in retained_ids)
        ]
        splits[split] = species_vectors(encoder, root, selected, classes, progress)
    del encoder
    gc.collect()
    progress({"stage": "training-head", "classCount": len(classes)})
    if incremental:
        from yizhu_server.plantnet_incremental import fit_additions

        head, optimization = fit_additions(
            splits, retained_parameters, len(additions), epochs=epochs,
            retained_replay=retained_replay,
            group_balanced_replay=group_balanced_replay,
        )
    else:
        head, optimization = fit_head(splits, len(classes), epochs=epochs)
    weights = output / "classifier.safetensors"
    save_file({key: value.detach().contiguous() for key, value in head.state_dict().items()}, weights)
    with torch.inference_mode():
        indices = head(splits["validation"][0]).topk(5, dim=1).indices
        labels = splits["validation"][1]
        validation = {
            "samples": len(labels),
            "top1": float(indices[:, 0].eq(labels).float().mean()),
            "top5": float(indices.eq(labels[:, None]).any(1).float().mean()),
        }
    del splits, head
    gc.collect()
    presence = PlantPresenceClassifier(presence_head, threads=threads)
    progress({"stage": "calibrating-gate"})
    # Include all snapshot validation species, not just already-trainable classes.
    gate = calibrate_gate(
        presence, binary_root, root, [row for row in rows if row["split"] == "validation"],
        progress,
    )
    if gate["headSha256"] != sha256(presence_head) or gate["binaryDatasetSha256"] != binary_digest:
        raise ValueError("Binary model or data changed during training.")
    write_report(output / "gate-validation.json", gate)
    model = {
        "createdAt": now(),
        "architecture": INCREMENTAL_ARCHITECTURE if incremental else ARCHITECTURE,
        "encoder": identity, "weightsSha256": sha256(weights),
        "classes": classes, "classCount": len(classes),
        "requestedClassCount": len(document["species"]), "splitTargets": targets,
        "snapshotSha256": sha256(root / "snapshot.json"),
        "samplesSha256": snapshot["samplesSha256"], "speciesSha256": snapshot["speciesSha256"],
        "presenceGate": {
            key: value for key, value in gate.items()
            if key not in {"binaryValidationCases", "speciesValidationCases"}
        },
        "gateValidationSha256": sha256(output / "gate-validation.json"),
        "planSha256": sha256(output / "plan.json"),
        "dataPolicy": document.get("policy"),
        "trainingScope": "Fixed Chinese regional-evidence candidates; actual classes listed",
        "dataLimitations": [
            *document.get("limitations", []),
            "Frozen public-image snapshot, not an independent final China field benchmark.",
            "Foundation pretraining overlap and prior experiment overlap are not ruled out.",
            "Binary presence does not reject out-of-set species or identify individual plants.",
            "Original Pl@ntNet checkpoint license needs deployment review; offline trial only.",
            "Cross-source photo duplicates checked, but cross-platform photographer identity unknown.",
        ],
        "training": optimization, "validation": validation,
        "trainingMode": "frozen-backbone-and-retained-rows" if incremental else "frozen-backbone",
        **({
            "retainedClassCount": len(retained), "trainedAdditionalClassCount": len(additions),
            "retainedSourceIndices": source_indices,
            "retainedParametersUnchanged": True,
            "retentionAccuracyGuaranteed": False,
            "retainedReplay": optimization["retainedReplay"],
        } if incremental else {}),
        "versions": {name: version(name) for name in ("torch", "torchvision", "safetensors")},
        "confidenceCalibrated": False, "productionApproved": False,
        "testUsedForSelection": False,
    }
    # Immutable fitted model and threshold exist BEFORE the separate evaluate command.
    write_report(output / "model.json", model)
    from yizhu_server.solution_metadata import write_artifact_metadata

    write_artifact_metadata(output / "model.json")
    progress({"stage": "trained-offline", "classes": len(classes), "validation": validation})
    return model


def ranked_candidates(logits, classes):
    import torch

    if logits.shape != (len(classes),) or not torch.isfinite(logits).all():
        raise ValueError("Invalid species logits.")
    order = torch.argsort(logits, descending=True, stable=True)[:5]
    scores = logits.softmax(0)
    return [{
        "speciesId": classes[i]["speciesId"], "scientificName": classes[i]["scientificName"],
        "chineseName": classes[i].get("chineseName"),
        "score": float(scores[i]), "confidence": None,
    } for i in order.tolist()]


class PlantNetCascade:
    def __init__(self, model_directory, *, presence_head=DEFAULT_PRESENCE, threads=1):
        import torch
        from safetensors.torch import load_file

        path = model_directory / "model.json"
        if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
            raise ValueError("Invalid model metadata size.")
        self.model = json.loads(path.read_text())
        model = self.model
        if (
            model.get("architecture") not in {ARCHITECTURE, INCREMENTAL_ARCHITECTURE}
            or model.get("confidenceCalibrated") is not False
            or model.get("classCount") != len(model["classes"])
            or sha256(model_directory / "classifier.safetensors") != model["weightsSha256"]
            or sha256(presence_head) != model["presenceGate"]["headSha256"]
            or sha256(model_directory / "gate-validation.json") != model["gateValidationSha256"]
            or sha256(model_directory / "plan.json") != model["planSha256"]
        ):
            raise ValueError("Model identity or pinned weights/threshold metadata changed.")
        validate_classes(model["classes"])
        self.threshold = model["presenceGate"]["threshold"]
        if (
            not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool)
            or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1
        ):
            raise ValueError("Invalid presence threshold.")
        incremental = model["architecture"] == INCREMENTAL_ARCHITECTURE
        self.encoder = SpeciesEncoder(threads=threads, normalize=not incremental)
        if self.encoder.identity != model["encoder"]:
            raise ValueError("Species encoder/preprocessing mismatch.")
        state = load_file(model_directory / "classifier.safetensors", device="cpu")
        if (
            set(state) != {"weight", "bias"}
            or state["weight"].shape != (model["classCount"], DIMENSION)
            or state["bias"].shape != (model["classCount"],)
            or any(not torch.isfinite(value).all() for value in state.values())
        ):
            raise ValueError("Invalid species head weights.")
        self.head = torch.nn.Linear(DIMENSION, model["classCount"])
        self.head.load_state_dict(state, strict=True)
        self.head.eval().requires_grad_(False)
        if incremental:
            from yizhu_server.plantnet_incremental import verify_retained_parameters

            verify_retained_parameters(self.encoder, self.head, model)
        self.presence = PlantPresenceClassifier(presence_head, threads=threads)
        if self.presence.encoder.identity != model["presenceGate"]["encoder"]:
            raise ValueError("Presence encoder/preprocessing mismatch.")
        self.version = sha256(path)

    def candidates(self, vector):
        import torch

        with torch.inference_mode():
            return ranked_candidates(self.head(vector), self.model["classes"])

    def predict(self, image):
        presence = self.presence.predict(image)
        passed = presence["score"] >= self.threshold
        return {
            "status": "requires_review" if passed else "nonplant",
            "presenceScore": presence["score"], "presenceThreshold": self.threshold,
            "candidates": self.candidates(self.encoder.encode(image)) if passed else [],
            "confidence": None, "modelVersion": self.version, "productionApproved": False,
            "scope": "closed-set-species-not-individual-identity",
        }


def summarize(cases, classes, requested_ids):
    """Never remove gate rejections, uncovered species or technical failures from denominator."""
    if not cases:
        raise ValueError("No plant test cases.")
    class_ids = {row["speciesId"] for row in classes}
    requested = set(requested_ids)
    if (
        not class_ids <= requested
        or len({row["sampleId"] for row in cases}) != len(cases)
        or any(row["speciesId"] not in requested for row in cases)
    ):
        raise ValueError("Duplicate cases or unexpected species in evaluation.")
    groups = {}
    for row in cases:
        groups.setdefault(row["speciesId"], []).append(row)

    def metrics(rows):
        def hit(row, key):
            return row["technicalSuccess"] and row["speciesId"] in class_ids and row[key]

        return {
            "samples": len(rows),
            "speciesTop1": sum(hit(r, "top1Hit") for r in rows) / len(rows),
            "speciesTop5": sum(hit(r, "top5Hit") for r in rows) / len(rows),
            "endToEndTop1": sum(r["gatePassed"] and hit(r, "top1Hit") for r in rows) / len(rows),
            "endToEndTop5": sum(r["gatePassed"] and hit(r, "top5Hit") for r in rows) / len(rows),
            "plantPassRate": sum(r["technicalSuccess"] and r["gatePassed"] for r in rows) / len(rows),
            "technicalFailures": sum(not r["technicalSuccess"] for r in rows),
        }

    per_class = {key: metrics(rows) for key, rows in sorted(groups.items())}
    covered = [row for row in cases if row["speciesId"] in class_ids]
    aggregate = metrics(cases)
    return {
        **aggregate,
        "macroEndToEndTop5": sum(r["endToEndTop5"] for r in per_class.values()) / len(per_class),
        "macroSpeciesTop5": sum(r["speciesTop5"] for r in per_class.values()) / len(per_class),
        "requestedClassCount": len(requested), "trainedClassCount": len(classes),
        "speciesWithTestPhotos": len(groups),
        "requestedSpeciesWithoutTestPhotos": sorted(requested - set(groups)),
        "uncoveredTestPhotos": len(cases) - len(covered),
        "coveredSpeciesOnly": metrics(covered) if covered else None,
        "perSpecies": per_class,
        "fullTargetCoverage": class_ids == requested and set(groups) == requested,
        # A public-cache test cannot certify independence or geographic field representativeness.
        "goalAchieved": False,
        "acceptanceBlockers": [
            *(["Incomplete target species/training/test coverage"] if (
                class_ids != requested or set(groups) != requested
            ) else []),
            *(["Observed end-to-end Top-5 below 90%"] if aggregate["endToEndTop5"] < .9 else []),
            "Independent China field benchmark not established",
        ],
    }


def check_external_holdout(document, rows, training_document, training_rows, binary_rows):
    """Audit all external test cases; reject contamination instead of shrinking the test."""
    def target_map(doc):
        return {r["speciesId"]: r["scientificName"] for r in doc["species"]}

    if target_map(document) != target_map(training_document):
        raise ValueError("External evaluation must preserve the full training target universe.")
    tests = [row for row in rows if row["split"] == "test"]
    if not tests:
        raise ValueError("External evaluation has no test photos.")
    exposed = [row for row in training_rows if row["split"] in {"train", "validation"}]
    identifiers = ("sampleId", "photoId", "observationId", "observerGroup")
    seen = {key: {row[key] for row in exposed} for key in identifiers}
    duplicates = DuplicateIndex()
    for row in exposed:
        duplicates.add(row["imageSha256"], int(row["differenceHash"], 16))
    for row in tests:
        if any(row[key] in seen[key] for key in identifiers) or duplicates.contains(
            row["imageSha256"], int(row["differenceHash"], 16),
        ):
            raise ValueError("External test overlaps species train/validation exposure.")
    check_cross_dataset_leakage(tests, binary_rows)
    return {
        "testPhotos": len(tests), "speciesTrainValidationPhotos": len(exposed),
        "checkedIdentityFields": list(identifiers),
        "exactAndNearDuplicateCheck": "sha256+dhash-distance<=4",
        "binaryTrainValidationImageOverlapChecked": True,
        "testCasesRemoved": 0, "independentFinalBenchmark": False,
        "limitations": [
            "Does not audit original backbone pretraining images.",
            "Cross-source photographer identities are not resolved.",
            "Reused development tests are not independent final acceptance data.",
        ],
    }


def evaluate(root, model_directory, output, *, presence_head=DEFAULT_PRESENCE,
             binary_root=DEFAULT_BINARY_DATA, threads=1, training_root=None):
    if output.exists():
        raise ValueError("Evaluation exists; do not overwrite a test result.")
    document, rows, snapshot = frozen_data(root)
    cascade = PlantNetCascade(model_directory, presence_head=presence_head, threads=threads)
    training_document, training_rows, training_snapshot = (
        frozen_data(training_root) if training_root is not None else (document, rows, snapshot)
    )
    if (
        cascade.model["snapshotSha256"] != sha256((training_root or root) / "snapshot.json")
        or cascade.model["samplesSha256"] != training_snapshot["samplesSha256"]
        or cascade.model["speciesSha256"] != training_snapshot["speciesSha256"]
    ):
        raise ValueError("Evaluation dataset differs from the frozen training snapshot.")
    binary_rows, digest = load_dataset(binary_root)
    if digest != cascade.model["presenceGate"]["binaryDatasetSha256"]:
        raise ValueError("Binary test dataset differs from calibration dataset.")
    holdout_audit = None
    if training_root is not None:
        holdout_audit = check_external_holdout(
            document, rows, training_document, training_rows, binary_rows,
        )
    cases = []
    for row in rows:
        if row["split"] != "test":
            continue
        case = {
            "sampleId": row["sampleId"], "speciesId": row["speciesId"],
            "technicalSuccess": False, "gatePassed": False, "top1Hit": False, "top5Hit": False,
        }
        try:
            presence = cascade.presence.predict_vector(feature(cascade.presence.encoder, root, row))
            # Evaluate the species head even for gate-rejected photos to measure gate impact.
            candidates = cascade.candidates(feature(cascade.encoder, root, row))
            ids = [item["speciesId"] for item in candidates]
            case.update({
                "technicalSuccess": True, "presenceScore": presence["score"],
                "gatePassed": presence["score"] >= cascade.threshold,
                "top1Hit": row["speciesId"] == ids[0], "top5Hit": row["speciesId"] in ids,
                "candidateSpeciesIds": ids,
            })
            retained_count = cascade.model.get("retainedClassCount", 0)
            if row["speciesId"] in {
                item["speciesId"] for item in cascade.model["classes"][:retained_count]
            }:
                import torch

                with torch.inference_mode():
                    logits = cascade.head(feature(cascade.encoder, root, row))[:retained_count]
                    ranks = logits.argsort(descending=True, stable=True)[:5].tolist()
                case["retainedOnlyTop5Hit"] = row["speciesId"] in [
                    cascade.model["classes"][i]["speciesId"] for i in ranks
                ]
        except (ValueError, RuntimeError) as exc:
            case["errorType"] = type(exc).__name__
        cases.append(case)
        if len(cases) % 25 == 0:
            print(f"Evaluated {len(cases)} species test photos", flush=True)
    scores, labels = [], []
    for row in binary_rows:
        if row["split"] == "test":
            result = cascade.presence.predict_vector(
                cached_feature(cascade.presence.encoder, binary_root, row),
            )
            scores.append(result["score"])
            labels.append(LABELS.index(row["label"]))
    report = {
        "createdAt": now(), "split": "test", "modelSha256": cascade.version,
        "snapshotSha256": sha256(root / "snapshot.json"),
        "trainingSnapshotSha256": cascade.model["snapshotSha256"],
        "evaluationMode": "external-development" if training_root is not None else "training-test",
        "externalHoldoutAudit": holdout_audit,
        "species": summarize(cases, cascade.model["classes"],
                             [row["speciesId"] for row in document["species"]]),
        "binary": binary_metrics(scores, labels, cascade.threshold),
        "presenceThreshold": cascade.threshold, "thresholdChangedUsingTest": False,
        "cases": cases, "independentFinalBenchmark": False, "productionApproved": False,
        "note": "Existing binary test has prior published trials; not a fresh final benchmark.",
    }
    if cascade.model.get("retainedClassCount"):
        a_ids = {
            item["speciesId"] for item in
            cascade.model["classes"][:cascade.model["retainedClassCount"]]
        }
        a_cases = [case for case in cases if case["speciesId"] in a_ids]
        b_cases = [case for case in cases if case["speciesId"] not in a_ids]
        report["retention"] = {
            "retainedClassCount": len(a_ids),
            "retainedSpeciesWithTestPhotos": len({case["speciesId"] for case in a_cases}),
            "retainedParametersUnchanged": True,
            "scope": "A-only logits baseline, not the full original 1081-output ranking",
            "retainedOnlyTop5": (
                sum(case.get("retainedOnlyTop5Hit", False) for case in a_cases) / len(a_cases)
            ) if a_cases else None,
            "combinedTop5OnA": (
                sum(case["top5Hit"] for case in a_cases) / len(a_cases)
            ) if a_cases else None,
            "endToEndTop5OnA": (
                sum(case["top5Hit"] and case["gatePassed"] for case in a_cases) / len(a_cases)
            ) if a_cases else None,
            "endToEndTop5OnB": (
                sum(case["top5Hit"] and case["gatePassed"] for case in b_cases) / len(b_cases)
            ) if b_cases else None,
            "aTestPhotos": len(a_cases), "bTestPhotos": len(b_cases),
        }
    write_report(output, report)
    return report


def compare_evaluations(root, baseline_directory, candidate_directory, output):
    """Paired development comparison, never automatic model selection or deployment."""
    if output.exists():
        raise ValueError("Comparison exists; do not overwrite it.")
    document, rows, snapshot = frozen_data(root)
    expected = {row["sampleId"]: row["speciesId"] for row in rows if row["split"] == "test"}
    if not expected:
        raise ValueError("No frozen test photos to compare.")
    models, reports, indexed = [], [], []
    for directory in (baseline_directory, candidate_directory):
        model = json.loads((directory / "model.json").read_text())
        report = json.loads((directory / "test.json").read_text())
        validate_classes(model["classes"])
        if (
            report.get("split") != "test"
            or report.get("modelSha256") != sha256(directory / "model.json")
            or report.get("snapshotSha256") != sha256(root / "snapshot.json")
            or model.get("snapshotSha256") != report["snapshotSha256"]
            or model.get("speciesSha256") != snapshot["speciesSha256"]
            or model.get("samplesSha256") != snapshot["samplesSha256"]
            or report.get("presenceThreshold") != model["presenceGate"]["threshold"]
            or report.get("thresholdChangedUsingTest") is not False
            or model.get("testUsedForSelection") is not False
            or model.get("classCount") != len(model["classes"])
            or not {r["speciesId"] for r in model["classes"]}.issubset(
                {r["speciesId"] for r in document["species"]}
            )
        ):
            raise ValueError("Comparison requires matching frozen data, model and pre-test settings.")
        cases = report["cases"]
        by_id = {case["sampleId"]: case for case in cases}
        if len(by_id) != len(cases) or {
            key: case["speciesId"] for key, case in by_id.items()
        } != expected:
            raise ValueError("Every frozen test photo must appear exactly once with its real label.")
        for case in cases:
            if any(type(case.get(key)) is not bool for key in (
                "technicalSuccess", "gatePassed", "top1Hit", "top5Hit",
            )) or (case["top1Hit"] and not case["top5Hit"]):
                raise ValueError("Invalid evaluation case flags.")
        models.append(model)
        reports.append(report)
        indexed.append(by_id)
    first, second = models
    if (
        [(r["speciesId"], r["scientificName"]) for r in first["classes"]]
        != [(r["speciesId"], r["scientificName"]) for r in second["classes"]]
        or first.get("retainedClassCount") != second.get("retainedClassCount")
        or first["encoder"] != second["encoder"]
        or any(first["presenceGate"][key] != second["presenceGate"][key]
               for key in ("headSha256", "encoder", "threshold"))
    ):
        raise ValueError("Paired head comparison requires identical classes, encoder and binary gate.")
    retained_count = first.get("retainedClassCount", 0)
    if type(retained_count) is not int or not 0 <= retained_count <= len(first["classes"]):
        raise ValueError("Invalid retained class count.")
    a_ids = {row["speciesId"] for row in first["classes"][:retained_count]}
    trained = {row["speciesId"] for row in first["classes"]}

    def hit(case):
        return (case["speciesId"] in trained and case["technicalSuccess"]
                and case["gatePassed"] and case["top5Hit"])

    def group(ids):
        before = [hit(indexed[0][key]) for key in ids]
        after = [hit(indexed[1][key]) for key in ids]
        return {
            "photos": len(ids), "speciesWithTestPhotos": len({expected[key] for key in ids}),
            "uncoveredPhotos": sum(expected[key] not in trained for key in ids),
            "baselineTop5Hits": sum(before), "candidateTop5Hits": sum(after),
            "baselineEndToEndTop5": sum(before) / len(ids) if ids else None,
            "candidateEndToEndTop5": sum(after) / len(ids) if ids else None,
            "improvedPhotos": sum(not b and a for b, a in zip(before, after, strict=True)),
            "regressedPhotos": sum(b and not a for b, a in zip(before, after, strict=True)),
            "baselineTechnicalFailures": sum(not indexed[0][key]["technicalSuccess"] for key in ids),
            "candidateTechnicalFailures": sum(not indexed[1][key]["technicalSuccess"] for key in ids),
        }

    comparison = {
        "createdAt": now(), "snapshotSha256": reports[0]["snapshotSha256"],
        "baselineModelSha256": reports[0]["modelSha256"],
        "candidateModelSha256": reports[1]["modelSha256"],
        "requestedClassCount": len(document["species"]), "trainedClassCount": len(trained),
        "retainedClassCount": retained_count,
        "all": group(list(expected)),
        "retainedA": group([key for key, species in expected.items() if species in a_ids]),
        "additionalBIncludingUncovered": group([
            key for key, species in expected.items() if species not in a_ids
        ]),
        "automaticSelection": False, "onlineChanged": False,
        "independentFinalBenchmark": False, "goalAchieved": False,
        "note": "Descriptive paired development test only. Keep all failures and uncovered photos; "
                "do not tune on this test or interpret partial coverage as final acceptance.",
    }
    write_report(output, comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "train", "evaluate", "predict", "compare"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--training-data", type=Path,
                        help="Original model snapshot; enables guarded external test evaluation")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--photo", type=Path)
    parser.add_argument("--presence-head", type=Path, default=DEFAULT_PRESENCE)
    parser.add_argument("--binary-data", type=Path, default=DEFAULT_BINARY_DATA)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--incremental", action="store_true",
                        help="Retain pretrained A parameters; train only ready B outputs")
    parser.add_argument("--retained-replay", action="store_true",
                        help="Use permitted A train photos to constrain new outputs (offline only)")
    parser.add_argument("--group-balanced-replay", action="store_true",
                        help="Fixed equal A/B training-loss mass; keep validation selection unchanged")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if args.training_data is not None and args.action != "evaluate":
        parser.error("--training-data is only valid for evaluate")
    if args.retained_replay and (args.action != "train" or not args.incremental):
        parser.error("--retained-replay requires train --incremental")
    if args.group_balanced_replay and (
        args.action != "train" or not args.incremental or not args.retained_replay
    ):
        parser.error("--group-balanced-replay requires train --incremental --retained-replay")
    required = {
        "snapshot": ("source", "data"), "train": ("data", "model"),
        "evaluate": ("data", "model", "output"), "predict": ("model", "photo"),
        "compare": ("data", "model", "baseline_model", "output"),
    }
    if any(getattr(args, key) is None for key in required[args.action]):
        parser.error(f"{args.action} requires {required[args.action]}")
    if args.action == "snapshot":
        result = freeze_dataset(args.source, args.data)
    elif args.action == "train":
        with dataset_lock(args.data):
            model = train(
                args.data, args.model, presence_head=args.presence_head,
                binary_root=args.binary_data, allow_partial=args.allow_partial,
                epochs=args.epochs, threads=args.threads,
                incremental=args.incremental, retained_replay=args.retained_replay,
                group_balanced_replay=args.group_balanced_replay,
            )
        result = {"classes": model["classCount"], "validation": model["validation"]}
    elif args.action == "evaluate":
        with dataset_lock(args.data):
            report = evaluate(
                args.data, args.model, args.output, presence_head=args.presence_head,
                binary_root=args.binary_data, threads=args.threads,
                training_root=args.training_data,
            )
        result = {key: value for key, value in report.items() if key != "cases"}
    elif args.action == "compare":
        result = compare_evaluations(args.data, args.baseline_model, args.model, args.output)
    else:
        cascade = PlantNetCascade(
            args.model, presence_head=args.presence_head, threads=args.threads,
        )
        with read_photo(args.photo) as image:
            result = cascade.predict(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
