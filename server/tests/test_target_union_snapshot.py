import hashlib

import pytest

from yizhu_server.target_union_snapshot import merge_rows

CLASSES = [
    {"speciesId": "one", "scientificName": "Example one", "chineseName": "甲"},
    {"speciesId": "two", "scientificName": "Example two", "chineseName": "乙"},
]


def sample(index, *, species="one", split="train", difference_hash=None):
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    return {
        "sampleId": f"sample:{index}", "photoId": f"photo:{index}",
        "observationId": f"observation:{index}", "speciesId": species,
        "scientificName": f"Example {'one' if species == 'one' else 'two'}",
        "split": split, "observerGroup": f"observer:{index}",
        "imageSha256": digest, "differenceHash": difference_hash or digest[:16],
    }


def test_primary_rows_are_kept_and_supplement_duplicates_are_skipped():
    row = sample(1)
    selected, rejected = merge_rows([row], [("old", [{**row}])], CLASSES)
    assert selected == [row]
    assert rejected["old:exact-or-near-duplicate"] == 1


def test_new_supplement_species_is_added_without_resplitting():
    old = sample(1)
    added = sample(2, species="two", split="validation")
    selected, rejected = merge_rows([old], [("old", [added])], CLASSES)
    assert selected == [old, added]
    assert selected[1]["split"] == "validation"
    assert not rejected


def test_identity_conflict_fails_closed():
    row = sample(1)
    conflicting = {**sample(2, species="two"), "photoId": row["photoId"]}
    with pytest.raises(ValueError, match="Conflicting sample identity"):
        merge_rows([row], [("old", [conflicting])], CLASSES)


def test_near_duplicate_is_not_added():
    row = sample(1, difference_hash="0000000000000000")
    near = sample(2, species="two", difference_hash="0000000000000001")
    selected, rejected = merge_rows([row], [("old", [near])], CLASSES)
    assert selected == [row]
    assert rejected["old:exact-or-near-duplicate"] == 1
