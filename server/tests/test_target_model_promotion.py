import json

import pytest

from yizhu_server.target_model_promotion import promote
from yizhu_server.training_data import sha256, write_json


def fixture(tmp_path, *, goal_met=True):
    model = tmp_path / "model"
    evaluation = tmp_path / "evaluation"
    model.mkdir()
    evaluation.mkdir()
    metadata = {
        "solutionId": "example", "status": "trained-offline-development-candidate",
        "productionApproved": False, "finalIndependentGoalMet": False,
        "plantSet": {"count": 2}, "classifier": {"sha256": "classifier"},
    }
    write_json(model / "model.metadata.json", metadata)
    result = {
        "status": "final-independent-evaluation", "goalMet": goal_met,
        "evaluatedOnce": True, "representedSpecies": 2,
        "modelMetadataSha256": sha256(model / "model.metadata.json"),
        "classifierSha256": "classifier", "macroEndToEndTop1": .81,
        "macroEndToEndTop5": .91, "productionChanged": False,
        "samples": 2, "top1": .81, "top5": .91, "macroTop1": .82,
        "macroTop5": .92, "plantPassRate": 1., "endToEndTop1": .81,
        "endToEndTop5": .91,
    }
    write_json(evaluation / "results.json", result)
    return model, evaluation


def test_promotion_requires_passing_independent_evidence(tmp_path):
    model, evaluation = fixture(tmp_path)
    result = promote(model, evaluation)
    assert result["status"] == "validated-independent"
    assert result["productionApproved"] is True
    assert json.loads((model / "production.metadata.json").read_text()) == result


def test_failed_independent_result_cannot_be_promoted(tmp_path):
    model, evaluation = fixture(tmp_path, goal_met=False)
    with pytest.raises(ValueError, match="incomplete or below"):
        promote(model, evaluation)


def test_production_metadata_cannot_be_overwritten(tmp_path):
    model, evaluation = fixture(tmp_path)
    (model / "production.metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        promote(model, evaluation)
