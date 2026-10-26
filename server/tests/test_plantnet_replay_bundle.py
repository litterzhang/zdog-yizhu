import json
from importlib.metadata import version

import pytest
from test_plantnet_transfer import classes, sample_dataset

from yizhu_server import plantnet_replay_bundle as bundle_module
from yizhu_server.training_data import sha256, write_json


@pytest.fixture
def bundle(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    rows = sample_dataset(tmp_path / "public-synthetic")
    root = tmp_path / "bundle"
    root.mkdir()
    provenance, tensors = {}, {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        labels = torch.tensor([int(r["speciesId"].split(":")[1]) for r in selected])
        x = torch.zeros(len(selected), 2048)
        x[torch.arange(len(selected)), labels] = 1.
        tensors[split + "_features"] = x
        tensors[split + "_labels"] = labels
        provenance[split] = selected
    tensors["retained_weight"] = torch.zeros(2, 2048)
    tensors["retained_weight"][0, 0] = tensors["retained_weight"][1, 1] = 5.
    tensors["retained_bias"] = torch.tensor([.1, -.1])
    save_file(tensors, root / "features.safetensors")
    write_json(root / "samples.json", provenance)
    plan = {
        "classes": classes(), "retainedClassCount": 2, "additionCount": 4,
        "splitCounts": {key: len(value) for key, value in provenance.items()},
        "testFeaturesIncluded": False, "privatePhotosUsed": False, "automaticDeployment": False,
        "seed": bundle_module.SEED, "epochs": 2,
        "variants": ["observed-class", "equal-groups"],
        "equalGroupMass": {"retainedA": .5, "additionalB": .5},
        "versions": {key: version(key) for key in ("torch", "torchvision", "safetensors")},
        "featuresSha256": sha256(root / "features.safetensors"),
        "samplesSha256": sha256(root / "samples.json"),
        "sourceSnapshotSha256": "synthetic-snapshot", "baselineModelSha256": "synthetic-model",
    }
    write_json(root / "plan.json", plan)
    return root, plan, tensors


@pytest.mark.parametrize("variant", ["observed-class", "equal-groups"])
def test_remote_head_bundle_trains_only_new_rows_and_records_fixed_variant(bundle, tmp_path, variant):
    import torch
    from safetensors.torch import load_file

    root, plan, original = bundle
    output = tmp_path / "trained"
    result = bundle_module.train_bundle(root, output, variant)
    assert result["classCount"] == 6
    assert result["variant"] == variant
    assert result["testUsedForSelection"] is False and result["automaticDeployment"] is False
    assert result["bundlePlanSha256"] == sha256(root / "plan.json")
    state = load_file(output / "classifier.safetensors")
    assert torch.equal(state["weight"][:2], original["retained_weight"])
    assert torch.equal(state["bias"][:2], original["retained_bias"])
    report = json.loads((output / "training.json").read_text())
    assert report["retainedReplay"]["groupBalanced"] is (variant == "equal-groups")
    assert report["retainedReplay"]["trainingLossMassA"] == pytest.approx(
        .5 if variant == "equal-groups" else 1 / 3,
    )
    with pytest.raises(ValueError, match="new output"):
        bundle_module.train_bundle(root, output, variant)


@pytest.mark.parametrize("change", ["versions", "checksum", "test-declared", "split", "labels", "tensor"])
def test_remote_bundle_rejects_mismatch_or_test_leak_before_creating_output(
    bundle, tmp_path, change,
):
    from safetensors.torch import save_file

    root, plan, tensors = bundle
    if change == "versions":
        plan["versions"]["torch"] = "different"
    elif change == "checksum":
        plan["featuresSha256"] = "invalid"
    elif change == "test-declared":
        plan["testFeaturesIncluded"] = True
    elif change == "split":
        rows = json.loads((root / "samples.json").read_text())
        rows["validation"][0]["split"] = "test"
        write_json(root / "samples.json", rows)
        plan["samplesSha256"] = sha256(root / "samples.json")
    else:
        if change == "labels":
            tensors["train_labels"][0] = 1
        else:
            tensors["test_features"] = tensors["validation_features"].clone()
        save_file(tensors, root / "features.safetensors")
        plan["featuresSha256"] = sha256(root / "features.safetensors")
    write_json(root / "plan.json", plan)
    with pytest.raises(ValueError):
        bundle_module.train_bundle(root, tmp_path / "rejected", "equal-groups")
    assert not (tmp_path / "rejected").exists()
