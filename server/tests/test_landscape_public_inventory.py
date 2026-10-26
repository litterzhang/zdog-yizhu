import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from yizhu_server import landscape_public_inventory as inventory
from yizhu_server.training_data import group_split

CLASSES = [
    {"speciesId": "one", "scientificName": "Example one", "chineseName": "合成甲"},
    {"speciesId": "two", "scientificName": "Example two", "chineseName": "合成乙"},
]


def sample(index, *, observer=None, split=None):
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    observer = observer or digest
    return {
        **CLASSES[0], "sampleId": str(index), "photoId": index, "observationId": index,
        "imageSha256": digest, "differenceHash": digest[:16],
        "observerGroup": observer, "split": split or group_split(observer),
        "license": "cc0", "attribution": "Synthetic fixture", "path": f"images/{index}.jpg",
    }


def candidates(rows):
    return [(r, Path(r["path"])) for r in rows]


def test_reuses_images_not_duplicate_copies_and_preserves_splits():
    row = sample(1)
    original = deepcopy(row)
    selected, skipped = inventory.select_reusable(candidates([row, deepcopy(row)]), CLASSES)
    assert [r for r, _ in selected] == [original]
    assert skipped["duplicateOrNearDuplicate"] == 1
    assert row == original


@pytest.mark.parametrize("field,value", [
    ("speciesId", "two"), ("split", "wrong"), ("observerGroup", "different"),
])
def test_conflicting_identity_fails_instead_of_changing_labels(field, value):
    row = sample(1)
    with pytest.raises(ValueError, match="conflicting"):
        inventory.select_reusable(candidates([row, {**row, field: value}]), CLASSES)


def test_nonpublic_license_and_path_escape_are_rejected():
    for change in ({"license": "cc-by-nc"}, {"path": "../user-image.jpg"}):
        with pytest.raises(ValueError):
            inventory.select_reusable(candidates([{**sample(1), **change}]), CLASSES)


def test_near_duplicates_and_observer_caps_are_enforced():
    rows = [sample(i, observer="a" * 64) for i in range(1, 6)]
    selected, rejected = inventory.select_reusable(candidates(rows), CLASSES)
    assert len(selected) == 3 and rejected["observerCap"] == 2
    rows = [sample(1), sample(2)]
    rows[1]["differenceHash"] = rows[0]["differenceHash"]
    selected, rejected = inventory.select_reusable(candidates(rows), CLASSES)
    assert len(selected) == 1 and rejected["duplicateOrNearDuplicate"] == 1


def test_images_of_species_outside_targets_are_not_imported():
    row = {**sample(1), "speciesId": "not-approved"}
    selected, _ = inventory.select_reusable(candidates([row]), CLASSES)
    assert selected == []


def test_split_quotas_cannot_be_filled_using_another_split(monkeypatch):
    monkeypatch.setattr(inventory, "TARGETS", {"train": 1, "validation": 1, "test": 1})
    rows = [sample(i, observer="a" * 64) for i in range(1, 4)]
    selected, rejected = inventory.select_reusable(candidates(rows), CLASSES)
    assert len(selected) == 1 and rejected["splitQuotaAlreadyMet"] == 2
