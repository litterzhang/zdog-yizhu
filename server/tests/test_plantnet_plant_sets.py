import pytest

from yizhu_server.plantnet_plant_sets import build_sets, pretrained_support_contract


def test_sets_are_disjoint_chinese_visible_and_preserve_pretrained_classes():
    def chinese(identifier, present=True, visible=True):
        return {
            "scientific_name": f"Test {identifier}", "chinese_name": "植物" if present else None,
            "aliases": "[]", "has_chinese_name": int(present), "is_visible": int(visible),
        }

    zh = {
        "existing": chinese("existing"), "both": chinese("both"), "new": chinese("new"),
        "empty": chinese("empty", False, False), "hidden": chinese("hidden", True, False),
    }
    records = [{
        "catalogSpeciesId": identifier, "classIndex": i, "sourceSpeciesId": str(i),
        "sourceScientificName": f"Test {identifier}",
    } for i, identifier in enumerate(["existing", "both", "empty", None])]
    candidates = [
        {"speciesId": k, "scientificName": f"Test {k}"}
        for k in ["both", "new", "empty", "hidden", "missing"]
    ]
    result = build_sets(records, zh, candidates)
    assert {r["speciesId"] for r in result["pretrainedSupported"]} == {"existing", "both"}
    assert {r["speciesId"] for r in result["finetuneAdditions"]} == {"new"}
    assert result["counts"]["pretrainedMappedToZhRowsRegardlessOfName"] == 3
    assert result["counts"]["combinedTarget"] == 3
    assert result["counts"]["existingOutsideEverydayCandidates"] == 1
    assert {r["reason"] for r in result["excludedCandidates"]} == {
        "no-species-zh-row", "no-chinese-name", "hidden",
    }
    assert result["pretrainedSupported"][0]["sourceClasses"][0]["classIndex"] == 1


def test_candidate_duplicate_ids_are_not_silently_collapsed():
    with pytest.raises(ValueError, match="Duplicate"):
        build_sets([], {}, [{"speciesId": "a"}, {"speciesId": "a"}])


def test_support_contract_never_promotes_planned_additions_or_overlap():
    source = {"labels.json": {"sha256": "pinned"}}
    existing = {
        "speciesId": "existing", "chineseName": "已有",
        "sourceClasses": [{"classIndex": 7}],
    }
    plan = {
        "sourceFiles": source, "pretrainedSupported": [existing],
        "finetuneAdditions": [{"speciesId": "not-trained"}],
        "candidateAlreadySupported": [existing],
        "counts": {"pretrainedChineseSupported": 1},
    }
    contract = pretrained_support_contract(plan, {"files": source})
    assert contract["supportedSpecies"] == [existing]
    assert "finetuneAdditions" not in contract
    plan["finetuneAdditions"].append({"speciesId": "another-plan"})
    assert pretrained_support_contract(plan, {"files": source}) == contract
    with pytest.raises(ValueError, match="bound"):
        pretrained_support_contract(plan, {"files": {}})
