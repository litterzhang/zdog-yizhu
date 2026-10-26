import json

import pytest

from yizhu_server.gbif_partial_finalize import finalize
from yizhu_server.training_data import sha256, write_json


def test_finalize_seals_verified_record_prefix(tmp_path):
    root = tmp_path / "data"
    (root / "records").mkdir(parents=True)
    (root / "images").mkdir()
    image = root / "images/a.jpg"
    image.write_bytes(b"image")
    row = {
        "candidateId": "one", "speciesId": "species",
        "path": "images/a.jpg", "imageSha256": sha256(image),
        "trainingEligible": False,
    }
    write_json(root / "records/one.json", row)
    write_json(root / "progress.json", {
        "stage": "collecting", "completedSpecies": 1, "totalSpecies": 2,
        "rejected": {}, "trainingEligible": False, "automaticAdmission": False,
    })
    names = tmp_path / "names.json"
    write_json(names, {"scientificNames": ["One", "Two"]})
    assert finalize(root, names) == [row]
    progress = json.loads((root / "progress.json").read_text())
    assert progress["stage"] == "collection-partial-finalized"
    assert progress["partialScope"] is True


def test_finalize_rejects_completed_or_unstarted_collection(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    write_json(root / "progress.json", {"stage": "collection-pass-finished"})
    names = tmp_path / "names.json"
    write_json(names, {"scientificNames": ["One"]})
    with pytest.raises(ValueError, match="in-progress"):
        finalize(root, names)
