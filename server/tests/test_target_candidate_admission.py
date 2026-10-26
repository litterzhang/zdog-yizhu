import hashlib
from pathlib import Path

from yizhu_server.target_candidate_admission import _normalized_hybrid_name, admit

CLASSES = [
    {"speciesId": "one", "scientificName": "Example one", "chineseName": "甲"},
    {"speciesId": "two", "scientificName": "Example two", "chineseName": "乙"},
]


def test_hybrid_marker_normalization_matches_plain_binomial():
    assert _normalized_hybrid_name("Begonia × hiemalis") == "Begonia hiemalis"


def row(index, *, species="one", split="validation", observer=None, dhash=None):
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    return {
        "sampleId": f"sample:{index}", "photoId": f"photo:{index}",
        "observationId": f"observation:{index}", "speciesId": species,
        "scientificName": f"Example {'one' if species == 'one' else 'two'}",
        "split": split, "observerGroup": observer or f"observer:{index}",
        "imageSha256": digest, "differenceHash": dhash or digest[:16],
    }


def test_missing_class_gets_one_whole_author_group_moved_to_train():
    base = [row(1, split="train")]
    candidates = [
        (row(2, species="two", observer="a"), Path("2.jpg"), "source"),
        (row(3, species="two", observer="a"), Path("3.jpg"), "source"),
        (row(4, species="two", observer="b"), Path("4.jpg"), "source"),
    ]
    admitted, rejected, overrides, counts = admit(base, candidates, CLASSES)
    assert not rejected
    assert len(overrides) == 1
    moved = [item for item, _, _ in admitted if item["observerGroup"] in overrides]
    assert moved and all(item["split"] == "train" for item in moved)
    assert counts["two"]["train"] == len(moved)


def test_candidate_near_duplicate_of_base_is_rejected():
    base = [row(1, split="train", dhash="0000000000000000")]
    candidate = row(2, species="two", dhash="0000000000000001")
    admitted, rejected, _, _ = admit(
        base, [(candidate, Path("2.jpg"), "source")], CLASSES,
    )
    assert admitted == []
    assert rejected["source:exact-or-near-duplicate"] == 1


def test_author_cap_is_applied_per_species():
    candidates = [
        (row(index, species="two", observer="same"), Path(f"{index}.jpg"), "source")
        for index in range(1, 8)
    ]
    admitted, rejected, _, _ = admit([], candidates, CLASSES)
    assert len(admitted) == 5
    assert rejected["source:author-cap"] == 2
