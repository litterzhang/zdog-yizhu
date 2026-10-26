import hashlib
import json
import random
from types import SimpleNamespace

import pytest
from PIL import Image

from yizhu_server import plantnet_transfer as transfer
from yizhu_server import training_data as data
from yizhu_server.solution_metadata import Exporter, artifact_metadata


def classes(count=6):
    return [
        {"speciesId": f"s:{i}", "scientificName": f"Test species{i}", "chineseName": f"植物{i}"}
        for i in range(count)
    ]


def sample_dataset(root):
    root.mkdir()
    (root / "images").mkdir()
    rows = []
    serial = 0
    for item in classes(7):
        for split in ("train", "validation", "test"):
            if item["speciesId"] == "s:6" and split != "test":
                continue  # A requested but not trainable class must not disappear from evaluation.
            while True:
                serial += 1
                group = data.observer_group(serial)
                if data.group_split(group) == split:
                    break
            path = root / "images" / f"{serial}.png"
            with Image.frombytes("RGB", (100, 100), random.Random(serial).randbytes(30000)) as image:
                image.save(path)
                dhash = data.difference_hash(image)
            rows.append({
                **item, "sampleId": f"photo:{serial}", "photoId": serial,
                "observationId": serial, "observerGroup": group, "split": split,
                "path": f"images/{serial}.png", "imageSha256": data.sha256(path),
                "differenceHash": f"{dhash:016x}", "license": "cc-by",
                "attribution": "Synthetic fixture, not a real accuracy benchmark",
            })
    data.write_json(root / "samples/collected.json", {"samples": rows})
    data.write_json(root / "species.json", {
        "species": classes(7), "splitTargets": {"train": 1, "validation": 1, "test": 1},
        "policy": "synthetic-test-only",
    })
    return rows


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    source, snapshot = tmp_path / "live", tmp_path / "snapshot"
    sample_dataset(source)
    transfer.freeze_dataset(source, snapshot)
    return source, snapshot


def test_snapshot_is_immutable_and_independent_of_running_collection(frozen):
    source, root = frozen
    document, rows, snapshot = transfer.frozen_data(root)
    assert len(document["species"]) == 7
    assert snapshot["readyClassCount"] == 6
    assert snapshot["requestedClassCount"] == 7 and snapshot["independentFinalBenchmark"] is False
    (source / rows[0]["path"]).write_bytes(b"changed only the live source")
    assert transfer.frozen_data(root)[1] == rows
    with pytest.raises(ValueError, match="exists"):
        transfer.freeze_dataset(source, root)


def test_snapshot_manifest_or_photo_tampering_is_rejected(frozen):
    _, root = frozen
    _, rows, _ = transfer.frozen_data(root)
    (root / rows[0]["path"]).write_bytes(b"bad bytes")
    with pytest.raises(ValueError, match="checksum"):
        transfer.frozen_data(root)
    data.write_json(root / "samples/extra.json", {"samples": []})
    with pytest.raises(ValueError, match="manifest"):
        transfer.frozen_data(root)


def test_snapshot_does_not_resume_paused_collection(tmp_path):
    data.write_json(tmp_path / "paused.json", {"reason": "user paused"})
    with pytest.raises(ValueError, match="paused"):
        transfer.freeze_dataset(tmp_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_conservative_gate_uses_both_validation_sources_and_never_gets_stricter():
    scores, labels = [.9, .8, .2, .1], [1, 1, 0, 0]
    assert transfer.conservative_threshold(scores, labels, [.95, .7], .5) == .5
    assert transfer.conservative_threshold(scores, labels, [.95, .05], .5) == .05
    assert transfer.conservative_threshold([.01, .9, .2], [1, 1, 0], [.8], .5) == .01
    with pytest.raises(ValueError):
        transfer.conservative_threshold(scores, labels, [float("nan")], .5)


def test_gate_rejections_uncovered_classes_and_failures_count_as_misses():
    base = {
        "technicalSuccess": True, "gatePassed": True, "top1Hit": True, "top5Hit": True,
    }
    cases = [
        {**base, "sampleId": "1", "speciesId": "s:0"},
        {**base, "sampleId": "2", "speciesId": "s:0", "gatePassed": False},
        {**base, "sampleId": "3", "speciesId": "s:6"},  # Not in the learned class set.
        {**base, "sampleId": "4", "speciesId": "s:1", "technicalSuccess": False},
    ]
    result = transfer.summarize(cases, classes(), [c["speciesId"] for c in classes(7)])
    assert result["speciesTop5"] == .5
    assert result["endToEndTop5"] == .25
    assert result["macroEndToEndTop5"] == pytest.approx(1 / 6)
    assert result["uncoveredTestPhotos"] == 1 and result["technicalFailures"] == 1
    assert not result["fullTargetCoverage"] and not result["goalAchieved"]
    assert len(result["requestedSpeciesWithoutTestPhotos"]) == 4


def test_perfect_small_closed_set_is_not_full_goal_completion():
    cases = [{
        "sampleId": str(i), "speciesId": row["speciesId"], "technicalSuccess": True,
        "gatePassed": True, "top1Hit": True, "top5Hit": True,
    } for i, row in enumerate(classes())]
    result = transfer.summarize(cases, classes(), [row["speciesId"] for row in classes()])
    assert result["fullTargetCoverage"] and result["endToEndTop5"] == 1
    assert result["goalAchieved"] is False
    assert result["acceptanceBlockers"] == ["Independent China field benchmark not established"]


def test_cross_dataset_duplicate_guard_checks_exact_and_near_duplicate():
    binary = [{"split": "train", "sha256": "a", "differenceHash": "0123456789abcdef"}]
    test_row = {"split": "test", "imageSha256": "a", "differenceHash": "fedcba9876543210"}
    with pytest.raises(ValueError, match="cross-split"):
        transfer.check_cross_dataset_leakage([test_row], binary)
    test_row.update(imageSha256="b", differenceHash="0123456789abcdea")
    with pytest.raises(ValueError, match="cross-split"):
        transfer.check_cross_dataset_leakage([test_row], binary)
    test_row["split"] = "train"
    transfer.check_cross_dataset_leakage([test_row], binary)


def test_ranking_keeps_stable_ids_and_does_not_claim_calibrated_confidence():
    torch = pytest.importorskip("torch")
    rows = transfer.ranked_candidates(torch.tensor([1., 4., 2., 6., 5., 3.]), classes())
    assert [r["speciesId"] for r in rows] == ["s:3", "s:4", "s:1", "s:5", "s:2"]
    assert all(r["confidence"] is None for r in rows)
    with pytest.raises(ValueError):
        transfer.ranked_candidates(torch.tensor([float("nan")] * 6), classes())


@pytest.mark.parametrize("incremental,retained_replay,group_balanced", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True),
])
def test_train_export_reload_and_two_stage_evaluation(
    frozen, tmp_path, monkeypatch, incremental, retained_replay, group_balanced,
):
    torch = pytest.importorskip("torch")
    _, root = frozen
    if incremental:
        document = json.loads((root / "species.json").read_text())
        for i, row in enumerate(document["species"]):
            row["classRole"] = "retained-pretrained" if i < 2 else "finetune-addition"
            if i < 2:
                row["sourceClasses"] = [{
                    "classIndex": i, "sourceSpeciesId": str(i),
                    "sourceScientificName": row["scientificName"],
                }]
        document["retainedClasses"] = document["species"][:2]
        document["retainedReplayAllowed"] = retained_replay
        data.write_json(root / "species.json", document)
        snapshot = json.loads((root / "snapshot.json").read_text())
        snapshot["speciesSha256"] = data.sha256(root / "species.json")
        data.write_json(root / "snapshot.json", snapshot)
    presence_path = tmp_path / "presence.json"
    presence_path.write_text("{}")
    presence_identity = {"weightsSha256": "synthetic-binary", "dimension": 2048}
    binary_rows = [{
        "id": f"{split}-{label}", "label": label, "split": split,
        "sha256": hashlib.sha256(f"{split}:{label}".encode()).hexdigest(),
        "differenceHash": hashlib.sha256(f"hash:{split}:{label}".encode()).hexdigest()[:16],
    } for split in ("validation", "test") for label in ("plant", "nonplant")]
    monkeypatch.setattr(transfer, "load_dataset", lambda _: (binary_rows, "binary-dataset"))

    class Encoder:
        identity = {"weightsSha256": "synthetic-species", "dimension": 2048}

        def __init__(self, *args, **kwargs):
            if incremental:
                self.original_head = torch.nn.Linear(2048, 2)
                with torch.no_grad():
                    self.original_head.weight.zero_()
                    self.original_head.bias.zero_()
                    self.original_head.weight[0, 0] = 5.
                    self.original_head.weight[1, 1] = 5.
                self.original_classes = [{
                    "sourceSpeciesId": str(i), "sourceScientificName": f"Test species{i}",
                } for i in range(2)]

    class Presence:
        def __init__(self, *args, **kwargs):
            self.encoder = SimpleNamespace(identity=presence_identity)
            self.version = data.sha256(presence_path)
            self.report = {
                "datasetSha256": "binary-dataset", "threshold": .5, "encoder": presence_identity,
            }

        def predict_vector(self, vector):
            return {"score": float(vector[0])}

    def feature(encoder, root, row):
        vector = torch.zeros(2048)
        if encoder.identity == presence_identity:
            vector[0] = .9
        else:
            vector[int(row["speciesId"].split(":")[1])] = 1.
        return vector

    def binary_feature(encoder, root, row):
        return torch.tensor([.9 if row["label"] == "plant" else .1])

    monkeypatch.setattr(transfer, "SpeciesEncoder", Encoder)
    monkeypatch.setattr(transfer, "PlantPresenceClassifier", Presence)
    monkeypatch.setattr(transfer, "feature", feature)
    monkeypatch.setattr(transfer, "cached_feature", binary_feature)
    seen_training_rows = []
    real_vectors = transfer.species_vectors

    def vectors(encoder, root, rows, classes, progress):
        assert all(row["split"] != "test" for row in rows)
        seen_training_rows.extend(row for row in rows if row["split"] == "train")
        return real_vectors(encoder, root, rows, classes, progress)

    monkeypatch.setattr(transfer, "species_vectors", vectors)
    model_dir = tmp_path / "models/transfer-v1"
    with pytest.raises(ValueError, match="allow-partial"):
        transfer.train(root, model_dir, presence_head=presence_path, epochs=30,
                       incremental=incremental, retained_replay=retained_replay,
                       group_balanced_replay=group_balanced)
    result = transfer.train(
        root, model_dir, presence_head=presence_path, allow_partial=True, epochs=60,
        incremental=incremental, retained_replay=retained_replay,
        group_balanced_replay=group_balanced,
    )
    assert result["classCount"] == 6 and result["requestedClassCount"] == 7
    assert result["validation"]["top1"] == 1 and not result["testUsedForSelection"]
    before = (model_dir / "model.json").read_bytes()
    report = transfer.evaluate(root, model_dir, tmp_path / "test.json", presence_head=presence_path)
    assert report["species"]["endToEndTop5"] == pytest.approx(6 / 7)
    assert report["species"]["uncoveredTestPhotos"] == 1
    assert not report["species"]["goalAchieved"]
    assert report["binary"]["falsePositiveRate"] == 0
    assert (model_dir / "model.json").read_bytes() == before
    external = tmp_path / "external"
    transfer.shutil.copytree(root, external)
    external_snapshot = json.loads((external / "snapshot.json").read_text())
    external_snapshot["createdAt"] = "synthetic-separate-snapshot"
    data.write_json(external / "snapshot.json", external_snapshot)
    with pytest.raises(ValueError, match="differs"):
        transfer.evaluate(external, model_dir, tmp_path / "rejected.json",
                          presence_head=presence_path)
    external_report = transfer.evaluate(
        external, model_dir, tmp_path / "external-test.json",
        presence_head=presence_path, training_root=root,
    )
    assert external_report["evaluationMode"] == "external-development"
    assert external_report["snapshotSha256"] != external_report["trainingSnapshotSha256"]
    assert external_report["trainingSnapshotSha256"] == data.sha256(root / "snapshot.json")
    assert external_report["species"] == report["species"]
    assert external_report["externalHoldoutAudit"]["testCasesRemoved"] == 0
    assert external_report["independentFinalBenchmark"] is False
    with pytest.raises(ValueError, match="differs"):
        transfer.evaluate(root, model_dir, tmp_path / "wrong-training.json",
                          presence_head=presence_path, training_root=external)
    assert not (tmp_path / "wrong-training.json").exists()
    metadata = artifact_metadata(model_dir / "model.json")
    assert metadata["plantSet"]["count"] == 6 and metadata["requestedClassCount"] == 7
    assert metadata["presenceGate"]["headSha256"] == data.sha256(presence_path)
    if incremental:
        assert metadata["retainedClassCount"] == 2
        assert metadata["trainedAdditionalClassCount"] == 4
        assert metadata["retentionAccuracyGuaranteed"] is False
        assert report["retention"]["retainedParametersUnchanged"] is True
        assert metadata["retainedReplay"]["enabled"] is retained_replay
        assert metadata["retainedReplay"]["groupBalanced"] is group_balanced
        retained_seen = {r["speciesId"] for r in seen_training_rows} & {"s:0", "s:1"}
        assert bool(retained_seen) is retained_replay
        if retained_replay:
            assert metadata["retainedReplay"]["trainingSpecies"] == 2
    with pytest.raises(ValueError, match="Evaluation exists"):
        transfer.evaluate(root, model_dir, tmp_path / "test.json")
    presence_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="identity"):
        transfer.PlantNetCascade(model_dir, presence_head=presence_path)


def test_replay_requires_explicit_snapshot_permission(frozen, tmp_path):
    _, root = frozen
    for incremental in (False, True):
        with pytest.raises(ValueError, match="explicitly allowed snapshot"):
            transfer.train(root, tmp_path / "not-created", incremental=incremental,
                           retained_replay=True)
        assert not (tmp_path / "not-created").exists()
    with pytest.raises(ValueError, match="Group-balanced replay requires"):
        transfer.train(root, tmp_path / "not-created", group_balanced_replay=True)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("field", [
    "sampleId", "photoId", "observationId", "observerGroup", "imageSha256", "differenceHash",
])
def test_external_holdout_rejects_any_training_or_calibration_overlap(frozen, split, field):
    _, root = frozen
    document, rows, _ = transfer.frozen_data(root)
    test = dict(next(row for row in rows if row["split"] == "test"))
    exposed = next(row for row in rows if row["split"] == split)
    test[field] = exposed[field]
    if field == "differenceHash":
        test[field] = f'{int(exposed[field], 16) ^ 1:016x}'
    with pytest.raises(ValueError, match="overlaps"):
        transfer.check_external_holdout(document, [test], document, rows, [])


def test_external_holdout_keeps_uncovered_species_and_rejects_target_changes(frozen):
    _, root = frozen
    document, rows, _ = transfer.frozen_data(root)
    audit = transfer.check_external_holdout(document, rows, document, rows, [])
    assert audit["testPhotos"] == 7  # seventh target lacks train/validation; never dropped
    assert audit["testCasesRemoved"] == 0
    assert audit["independentFinalBenchmark"] is False
    altered = {**document, "species": document["species"][:-1]}
    with pytest.raises(ValueError, match="full training target"):
        transfer.check_external_holdout(altered, rows, document, rows, [])
    renamed = {**document, "species": [
        {**row, "scientificName": "Different name"} for row in document["species"]
    ]}
    with pytest.raises(ValueError, match="full training target"):
        transfer.check_external_holdout(renamed, rows, document, rows, [])
    with pytest.raises(ValueError, match="no test"):
        transfer.check_external_holdout(document, [], document, rows, [])


@pytest.mark.parametrize("split", ["train", "validation"])
def test_external_holdout_rejects_binary_exposure(frozen, split):
    _, root = frozen
    document, rows, _ = transfer.frozen_data(root)
    test = next(row for row in rows if row["split"] == "test")
    binary = [{
        "split": split, "sha256": test["imageSha256"],
        "differenceHash": test["differenceHash"],
    }]
    with pytest.raises(ValueError, match="cross-split"):
        transfer.check_external_holdout(document, rows, document, rows, binary)


@pytest.fixture
def paired_reports(frozen, tmp_path):
    _, root = frozen
    _, rows, snapshot = transfer.frozen_data(root)
    directories = [tmp_path / "baseline", tmp_path / "candidate"]
    for index, directory in enumerate(directories):
        model = {
            "classes": classes(), "classCount": 6, "retainedClassCount": 2,
            "encoder": {"synthetic": True},
            "presenceGate": {"headSha256": "same", "encoder": {}, "threshold": .1},
            "snapshotSha256": data.sha256(root / "snapshot.json"),
            "speciesSha256": snapshot["speciesSha256"], "samplesSha256": snapshot["samplesSha256"],
            "testUsedForSelection": False, "syntheticTrial": index,
        }
        data.write_json(directory / "model.json", model)
        cases = []
        for row in rows:
            if row["split"] != "test":
                continue
            hit = row["speciesId"] not in ({"s:1", "s:6"} if index == 0 else {"s:0", "s:2", "s:6"})
            cases.append({
                "sampleId": row["sampleId"], "speciesId": row["speciesId"],
                "technicalSuccess": not (index == 1 and row["speciesId"] == "s:2"),
                "gatePassed": True, "top1Hit": hit, "top5Hit": hit,
            })
        data.write_json(directory / "test.json", {
            "split": "test", "cases": list(reversed(cases)) if index else cases,
            "modelSha256": data.sha256(directory / "model.json"),
            "snapshotSha256": model["snapshotSha256"],
            "presenceThreshold": .1, "thresholdChangedUsingTest": False,
        })
    return root, *directories, tmp_path / "comparison.json"


def test_paired_comparison_counts_uncovered_failures_and_both_directions(paired_reports):
    result = transfer.compare_evaluations(*paired_reports)
    assert result["all"]["photos"] == 7
    assert result["all"]["uncoveredPhotos"] == 1
    assert result["all"]["baselineTop5Hits"] == 5
    assert result["all"]["candidateTop5Hits"] == 4
    assert result["all"]["improvedPhotos"] == 1
    assert result["all"]["regressedPhotos"] == 2
    assert result["all"]["candidateTechnicalFailures"] == 1
    assert result["retainedA"]["photos"] == 2
    assert result["retainedA"]["improvedPhotos"] == result["retainedA"]["regressedPhotos"] == 1
    assert result["additionalBIncludingUncovered"]["photos"] == 5
    assert result["automaticSelection"] is False
    assert result["goalAchieved"] is False
    with pytest.raises(ValueError, match="overwrite"):
        transfer.compare_evaluations(*paired_reports)


@pytest.mark.parametrize("change", [
    "missing-photo", "duplicate-photo", "wrong-label", "wrong-model", "changed-threshold",
    "changed-classes", "changed-snapshot", "invalid-flags",
])
def test_incomparable_results_cannot_be_reported_as_improvement(paired_reports, change):
    root, baseline, candidate, output = paired_reports
    report = json.loads((candidate / "test.json").read_text())
    if change == "missing-photo":
        report["cases"].pop()
    elif change == "duplicate-photo":
        report["cases"].append(report["cases"][0])
    elif change == "wrong-label":
        report["cases"][0]["speciesId"] = "wrong"
    elif change == "wrong-model":
        report["modelSha256"] = "not-the-model"
    elif change == "changed-threshold":
        report["presenceThreshold"] = .9
    elif change == "changed-snapshot":
        report["snapshotSha256"] = "different"
    elif change == "invalid-flags":
        report["cases"][0]["technicalSuccess"] = "true"
    else:
        model = json.loads((candidate / "model.json").read_text())
        model["classes"] = list(reversed(model["classes"]))
        data.write_json(candidate / "model.json", model)
        report["modelSha256"] = data.sha256(candidate / "model.json")
    data.write_json(candidate / "test.json", report)
    with pytest.raises(ValueError):
        transfer.compare_evaluations(root, baseline, candidate, output)
    assert not output.exists()


def test_new_architecture_is_exported_without_renaming_existing_bioclip_models(tmp_path, monkeypatch):
    from yizhu_server.catalog import PlantCatalog
    from yizhu_server.solution_metadata import write_artifact_metadata

    path = tmp_path / "server/.artifacts/models/plantnet-china-v1/model.json"
    data.write_json(path, {
        "architecture": transfer.ARCHITECTURE, "encoder": {}, "weightsSha256": "test",
        "classes": classes(), "classCount": 6, "requestedClassCount": 972,
        "trainingMode": "frozen-backbone", "presenceGate": {"threshold": .1},
    })
    write_artifact_metadata(path)
    exporter = Exporter(tmp_path / "metadata", PlantCatalog(None))
    index = exporter.export(tmp_path / "server")
    assert index["solutions"]["plantnet-resnet50-linear-plantnet-china-v1"]["plantCount"] == 6
    assert not any(key.startswith("bioclip2-linear-plantnet") for key in index["solutions"])
