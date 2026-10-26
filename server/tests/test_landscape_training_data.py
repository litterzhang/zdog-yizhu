import json

import pytest

from yizhu_server import landscape_training_data as landscape
from yizhu_server.training_data import write_json


def targets():
    return {
        "level": "species", "unresolvedInputCount": 16,
        "classes": [
            {"speciesId": "a", "scientificName": "Osmanthus fragrans", "chineseName": "木犀"},
            {"speciesId": "b", "scientificName": "Ginkgo biloba", "chineseName": "银杏"},
            {"speciesId": "c", "scientificName": "Other species", "chineseName": "其他"},
        ],
    }


def test_prepare_preserves_only_resolved_species(tmp_path):
    task, root = tmp_path / "task", tmp_path / "data"
    write_json(task / "targets.json", targets())
    document = landscape.prepare(root, task)
    assert document["species"] == targets()["classes"]
    assert document["unresolvedInputCount"] == 16
    assert "trainedSpecies" not in document
    assert landscape.prepare(root, task) == document
    changed = targets()
    changed["classes"].pop()
    write_json(task / "targets.json", changed)
    with pytest.raises(ValueError, match="scope changed"):
        landscape.prepare(root, task)


@pytest.mark.parametrize("mutation", ["duplicate", "empty-name", "not-species"])
def test_prepare_rejects_unsafe_scope(tmp_path, mutation):
    document = targets()
    if mutation == "duplicate":
        document["classes"].append(document["classes"][0])
    elif mutation == "empty-name":
        document["classes"][0]["chineseName"] = ""
    else:
        document["level"] = "cultivar"
    write_json(tmp_path / "task/targets.json", document)
    with pytest.raises(ValueError, match="approved species"):
        landscape.prepare(tmp_path / "data", tmp_path / "task")


def test_schedule_skips_ready_and_prioritizes_osmanthus_without_mutating_scope():
    document = {"species": targets()["classes"]}
    original = json.dumps(document)
    rows = [
        {"speciesId": "c", "split": split}
        for split, count in landscape.TARGETS.items() for _ in range(count)
    ] + [{"speciesId": "b", "split": "train"}] * 60
    assert [r["speciesId"] for r in landscape.schedule(document, rows)["species"]] == ["a", "b"]
    assert json.dumps(document) == original
    # Missing validation/test cannot be substituted with more training images.
    assert "b" in [r["speciesId"] for r in landscape.schedule(document, rows)["species"]]


def test_seed_refuses_changed_manifest_or_collection_already_started(tmp_path):
    root, task = tmp_path / "data", tmp_path / "task"
    write_json(task / "verified-image-files.json", {"files": []})
    write_json(root / "species.json", {})
    write_json(root / "collection.json", {})
    with pytest.raises(ValueError, match="after collection"):
        landscape.seed(root, task, {})
    write_json(root / "seed-import.json", {"inventorySha256": "wrong"})
    with pytest.raises(ValueError, match="inputs changed"):
        landscape.seed(root, task, {})


def test_deferred_species_remain_shortfalls_not_ready(tmp_path):
    document = {
        "species": targets()["classes"], "deferredSpeciesIds": ["a"],
        "unresolvedInputCount": 16,
    }
    assert len(landscape.schedule(document, [])["species"]) == 3
    assert [r["speciesId"] for r in landscape.collection_schedule(document, [])["species"]] == [
        "b", "c",
    ]
    report = landscape.overall_coverage(tmp_path, document)
    assert report["approvedSpecies"] == 3 and report["readySpeciesCount"] == 0
    assert len(report["shortfalls"]) == 3
    assert report["shortfalls"][0]["deferredFromThisSourcePass"] is True
    assert report["trainedSupportUpdated"] is False


def test_collection_deferrals_are_pinned_to_the_target_scope(tmp_path):
    from yizhu_server.training_data import sha256

    root, task = tmp_path / "data", tmp_path / "task"
    write_json(task / "targets.json", targets())
    policy = {
        "targetManifestSha256": sha256(task / "targets.json"), "deferredSpeciesIds": ["a"],
    }
    write_json(task / "collection-policy.json", policy)
    assert landscape.prepare(root, task)["deferredSpeciesIds"] == ["a"]
    policy["deferredSpeciesIds"] = ["other"]
    write_json(task / "collection-policy.json", policy)
    with pytest.raises(ValueError, match="Deferred collection scope"):
        landscape.prepare(tmp_path / "new-data", task)
