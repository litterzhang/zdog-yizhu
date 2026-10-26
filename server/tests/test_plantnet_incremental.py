from types import SimpleNamespace

import pytest

from yizhu_server.plantnet_incremental import (
    fit_additions,
    replay_class_weights,
    retained_parameters_for,
    verify_retained_parameters,
)


def test_training_updates_only_new_rows_and_old_logits_remain_identical():
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[5., 0., 0., 0.], [0., 5., 0., 0.]])
    bias = torch.tensor([.1, -.1])
    x = torch.eye(4)
    splits = {
        "train": (x[2:].repeat(20, 1), torch.tensor([2, 3] * 20)),
        "validation": (x, torch.arange(4)),
    }
    head, history = fit_additions(splits, (weight, bias), 2, epochs=500)
    assert torch.equal(head.weight[:2].detach(), weight)
    assert torch.equal(head.bias[:2].detach(), bias)
    assert torch.equal(head(x)[:, :2], torch.nn.functional.linear(x, weight, bias))
    assert head(x).argmax(1).tolist() == [0, 1, 2, 3]
    assert history["testUsedForSelection"] is False
    assert history["retainedValidationSpecies"] == 2


def test_original_A_photos_cannot_become_new_training_labels():
    torch = pytest.importorskip("torch")
    x = torch.eye(4)
    with pytest.raises(ValueError, match="Train only B"):
        fit_additions({
            "train": (x, torch.arange(4)), "validation": (x, torch.arange(4)),
        }, (x[:2], torch.zeros(2)), 2, epochs=1)


@pytest.mark.parametrize("group_balanced", [False, True])
def test_replay_constrains_new_outputs_without_updating_old_parameters(group_balanced):
    torch = pytest.importorskip("torch")
    x = torch.eye(4)
    weight = x[:2] * 5
    bias = torch.tensor([.1, -.1])
    # Unequal class counts exercise class-balanced replay, not label duplication.
    labels = torch.tensor([0] * 3 + [1] * 5 + [2] * 20 + [3] * 20)
    head, report = fit_additions({
        "train": (x[labels], labels), "validation": (x, torch.arange(4)),
    }, (weight, bias), 2, epochs=500, retained_replay=True,
        group_balanced_replay=group_balanced)
    assert torch.equal(head.weight[:2].detach(), weight)
    assert torch.equal(head.bias[:2].detach(), bias)
    assert head(x).argmax(1).tolist() == [0, 1, 2, 3]
    assert report["retainedReplay"]["trainingPhotos"] == 8
    assert report["retainedReplay"]["trainingSpecies"] == 2
    assert report["retainedReplay"]["allRetainedSpeciesCovered"]
    assert report["retainedReplay"]["loss"] == (
        "equal-group-class-balanced-cross-entropy" if group_balanced
        else "observed-class-balanced-cross-entropy"
    )
    assert report["retainedReplay"]["groupBalanced"] is group_balanced
    assert report["retainedReplay"]["trainingLossMassA"] == pytest.approx(.5)
    assert report["testUsedForSelection"] is False
    assert report["validationSelection"] == "macro cross entropy over observed A/B validation species"


@pytest.mark.parametrize("additions", [3, 36, 172])
def test_group_balance_keeps_equal_mass_as_new_class_count_grows(additions):
    torch = pytest.importorskip("torch")
    # Two observed A classes; absent A classes get no fabricated training rows.
    labels = torch.tensor([0] * 3 + [2] * 7 + [
        label for label in range(4, 4 + additions) for _ in range(label)
    ])
    weights = replay_class_weights(labels, 4, additions, group_balanced=True)
    contributions = torch.bincount(labels, minlength=4 + additions) * weights / len(labels)
    assert float(contributions[:4].sum()) == pytest.approx(.5)
    assert float(contributions[4:].sum()) == pytest.approx(.5)
    assert contributions[1] == contributions[3] == 0
    assert float(contributions[0]) == pytest.approx(.25)
    assert float(contributions[-1]) == pytest.approx(.5 / additions)
    ordinary = replay_class_weights(labels, 4, additions)
    ordinary_mass = float(ordinary[labels[labels < 4]].sum()) / len(labels)
    assert ordinary_mass == pytest.approx(2 / (2 + additions))


def test_group_balance_requires_explicit_replay_and_real_old_training_rows():
    torch = pytest.importorskip("torch")
    x = torch.eye(4)
    with pytest.raises(ValueError, match="requires retained replay"):
        fit_additions({
            "train": (x[2:], torch.tensor([2, 3])), "validation": (x, torch.arange(4)),
        }, (x[:2], torch.zeros(2)), 2, epochs=1, group_balanced_replay=True)
    with pytest.raises(ValueError, match="observed A"):
        replay_class_weights(torch.tensor([2, 3]), 2, 2, group_balanced=True)


def test_replay_cannot_substitute_validation_for_missing_training_photos():
    torch = pytest.importorskip("torch")
    x = torch.eye(4)
    with pytest.raises(ValueError, match="A train-split"):
        fit_additions({
            "train": (x[2:], torch.tensor([2, 3])),
            "validation": (x, torch.arange(4)),
        }, (x[:2], torch.zeros(2)), 2, epochs=1, retained_replay=True)
    with pytest.raises(ValueError, match="every B"):
        fit_additions({
            "train": (x[:3], torch.tensor([0, 1, 2])),
            "validation": (x, torch.arange(4)),
        }, (x[:2], torch.zeros(2)), 2, epochs=1, retained_replay=True)


def test_retained_indices_and_parameter_identity_are_verified_on_load():
    torch = pytest.importorskip("torch")
    original = torch.nn.Linear(4, 3)
    encoder = SimpleNamespace(
        original_head=original,
        original_classes=[
            {"sourceSpeciesId": str(i), "sourceScientificName": f"Test species{i}"}
            for i in range(3)
        ],
    )
    classes = [{
        "sourceClasses": [{
            "classIndex": 2, "sourceSpeciesId": "2", "sourceScientificName": "Test species2",
        }],
    }]
    (weight, bias), indices = retained_parameters_for(encoder, classes)
    assert indices == [2]
    head = torch.nn.Linear(4, 2)
    with torch.no_grad():
        head.weight[0].copy_(weight[0])
        head.bias[0].copy_(bias[0])
    meta = {"retainedClassCount": 1, "trainedAdditionalClassCount": 1, "classCount": 2,
            "classes": classes + [{}], "retainedSourceIndices": [2]}
    verify_retained_parameters(encoder, head, meta)
    with torch.no_grad():
        head.bias[0] += .01
    with pytest.raises(ValueError, match="parameters changed"):
        verify_retained_parameters(encoder, head, meta)
    classes[0]["sourceClasses"][0]["sourceScientificName"] = "Wrong species"
    with pytest.raises(ValueError, match="differs"):
        retained_parameters_for(encoder, classes)
