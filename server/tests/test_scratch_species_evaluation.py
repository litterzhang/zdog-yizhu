import hashlib
from copy import deepcopy

import pytest

from yizhu_server.scratch_species_evaluation import summarize, validate_inputs
from yizhu_server.trained_species_contract import build_trained_contract
from yizhu_server.training_data import group_split


def inputs():
    classes = [
        {"speciesId": str(i), "scientificName": f"Example species{i}", "chineseName": f"合成{i}"}
        for i in range(2)
    ]
    samples = []
    serial = 0
    for species in classes:
        for split in ("train", "validation", "test"):
            while True:
                serial += 1
                digest = hashlib.sha256(str(serial).encode()).hexdigest()
                if group_split(digest) == split:
                    break
            samples.append({
                **species, "sampleId": f"example:{serial}", "observationId": serial,
                "photoId": serial, "observerGroup": digest, "imageSha256": digest,
                "differenceHash": digest[:16], "path": f"images/{serial}.jpg",
                "license": "cc0", "attribution": "Synthetic fixture", "split": split,
            })
    contract = build_trained_contract(
        classes, target_species_ids=["0", "1"], training_examples={"0": 1, "1": 1},
        completed_epochs=2, backbone_optimizer_steps=2, checkpoint_sha256="a" * 64,
    )
    training = {
        "classes": classes, "samples": [r for r in samples if r["split"] != "test"],
        "initialization": "random", "pretrainedWeightsAllowed": False,
        "pretrainedFeaturesAllowed": False, "testImagesIncluded": False,
        "targetSpeciesIds": ["0", "1"],
    }
    heldout = {
        "classes": classes, "samples": [r for r in samples if r["split"] == "test"],
        "trainingManifestSha256": "b" * 64,
    }
    metadata = {
        "classes": contract["classes"], "classCount": 2, "status": "trained-offline",
        "initialization": "random", "pretrainedWeightsUsed": False,
        "pretrainedFeaturesUsed": False, "inheritedClassCount": 0, "trainBackbone": True,
        "backboneChangedFromRandomInitialization": True, "testUsedForSelection": False,
        "automaticDeployment": False, "externalSpeciesFallbackAllowed": False,
        "weightsSha256": "a" * 64, "contractSha256": contract["contractSha256"],
        "manifestSha256": "b" * 64,
    }
    return metadata, contract, training, heldout, {
        "classes": classes + [{"speciesId": "2", "scientificName": "Untrained species"}],
        "unresolvedInputCount": 1,
    }


def check(args):
    metadata, contract, training, heldout, targets = args
    return validate_inputs(metadata, contract, training, heldout,
                           training_sha="b" * 64, targets=targets)


def test_frozen_manifests_and_clean_model_validate_without_mutation():
    args = inputs()
    before = deepcopy(args)
    assert len(check(args)) == 2
    assert args == before


@pytest.mark.parametrize("field,value", [
    ("pretrainedWeightsUsed", True), ("testUsedForSelection", True),
    ("automaticDeployment", True), ("externalSpeciesFallbackAllowed", True),
    ("manifestSha256", "different"), ("weightsSha256", "different"),
])
def test_rejects_mixed_models_or_test_selection(field, value):
    args = inputs()
    args[0][field] = value
    with pytest.raises(ValueError):
        check(args)


def test_rejects_label_reordering_and_whitelist_mismatch():
    args = inputs()
    args[3]["classes"] = list(reversed(args[3]["classes"]))
    with pytest.raises(ValueError, match="class order"):
        check(args)
    args = inputs()
    args[4]["classes"][0] = {"speciesId": "0", "scientificName": "Different species"}
    with pytest.raises(ValueError, match="target identities"):
        check(args)


def test_no_duplicate_test_images_or_training_samples_in_holdout():
    args = inputs()
    args[3]["samples"].append(deepcopy(args[3]["samples"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        check(args)
    args = inputs()
    args[3]["samples"][0] = deepcopy(args[2]["samples"][0])
    with pytest.raises(ValueError):
        check(args)


def test_missing_test_species_cannot_be_silently_skipped():
    args = inputs()
    args[3]["samples"].pop()
    with pytest.raises(ValueError, match="Every trained species"):
        check(args)


def predictions():
    return [
        {"sampleId": "a", "expectedSpeciesId": "0", "rankedSpeciesIds": ["0", "1"]},
        {"sampleId": "b", "expectedSpeciesId": "0", "rankedSpeciesIds": ["1", "0"]},
        {"sampleId": "c", "expectedSpeciesId": "1", "rankedSpeciesIds": ["1", "0"]},
    ]


def test_metrics_and_target_coverage_are_not_an_end_to_end_claim():
    _, contract, _, _, targets = inputs()
    result = summarize(predictions(), contract["classes"], targets)
    assert result["sampleCount"] == 3 and result["top1Correct"] == 2
    assert result["top1"] == pytest.approx(2 / 3)
    assert result["macroTop1"] == .75 and result["top5"] == 1
    assert result["topConfusions"] == [
        {"expectedSpeciesId": "0", "predictedSpeciesId": "1", "count": 1}
    ]
    assert result["targetCoverage"]["unevaluatedSpeciesIds"] == ["2"]
    assert result["endToEndTop5"] is None and result["goalAchieved"] is False
    assert result["independentFinalBenchmark"] is False


@pytest.mark.parametrize("change", ["duplicate", "external", "truncated", "missing-class"])
def test_invalid_predictions_never_produce_partial_accuracy(change):
    _, contract, _, _, targets = inputs()
    rows = predictions()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "external":
        rows[0]["rankedSpeciesIds"] = ["0", "2"]
    elif change == "truncated":
        rows[0]["rankedSpeciesIds"] = ["0"]
    else:
        rows.pop()
    with pytest.raises(ValueError):
        summarize(rows, contract["classes"], targets)
