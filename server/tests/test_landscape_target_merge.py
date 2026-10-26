import runpy
from copy import deepcopy
from pathlib import Path

import pytest

merge = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/merge-landscape-targets.py")
)["merge"]


def fixtures():
    base = {
        "inputRecordCount": 2, "linkedInputCount": 1,
        "classes": [{
            "speciesId": "a", "scientificName": "Existing species", "chineseName": "原名",
            "sourceInputs": [{"inputName": "原名", "articleNumber": 1}],
        }],
        "unresolvedInputs": [{"inputName": "旧歧义名", "note": "尚未确定", "articleNumber": 2}],
    }
    records = [
        {"speciesId": key, "scientificName": scientific, "chineseName": name,
         "displayName": name, "aliases": aliases, "isVisible": True}
        for key, scientific, name, aliases in [
            ("a", "Existing species", "原名", ["别名"]),
            ("b", "New species", "新名", []),
        ]
    ]
    return base, records, {"scientificNameLinks": {}, "needsReview": {}}


def test_stable_id_deduplication_retains_every_input_and_original_pending():
    base, records, notes = fixtures()
    before = deepcopy(base)
    result = merge(base, ["原名", "别名", "新名", "未匹配"], notes, records)
    assert result["resolvedUniqueSpeciesCount"] == 2
    assert result["inputRecordCount"] == 6
    assert result["linkedInputCount"] == 4
    assert result["unresolvedInputCount"] == 2
    assert result["summary"]["newDistinctSpecies"] == 1
    assert result["summary"]["overlappingAdditionalNames"] == 2
    assert base == before
    assert result["trainedSupportDeclared"] is False
    assert result["runningCollectionOrTrainingChanged"] is False


def test_review_block_takes_precedence_over_exact_name():
    base, records, notes = fixtures()
    notes["needsReview"]["新名"] = "同名有歧义"
    result = merge(base, ["新名"], notes, records)
    assert result["summary"]["newDistinctSpecies"] == 0
    assert result["additionalInputMappings"][0]["speciesId"] is None
    assert result["additionalInputMappings"][0]["candidates"][0]["speciesId"] == "b"


def test_scoped_link_does_not_create_global_alias_or_modify_catalog():
    base, records, notes = fixtures()
    before = deepcopy(records)
    notes["scientificNameLinks"]["园艺名"] = {
        "scientificName": "New species", "note": "限定本名单", "sources": ["reference"],
    }
    result = merge(base, ["园艺名"], notes, records)
    assert result["additionalInputMappings"][0]["speciesId"] == "b"
    assert records == before


def test_ambiguous_and_hidden_catalog_matches_remain_pending():
    base, records, notes = fixtures()
    records[1]["aliases"] = ["原名"]
    records[1]["isVisible"] = False
    result = merge(base, ["原名", "新名"], notes, records)
    assert result["summary"]["pendingAdditionalNames"] == 2


def test_changed_base_identity_is_rejected():
    base, records, notes = fixtures()
    records[0]["scientificName"] = "Different species"
    with pytest.raises(ValueError, match="identity differs"):
        merge(base, [], notes, records)


def test_reviewed_hidden_species_can_join_plan_without_changing_database():
    base, records, notes = fixtures()
    records[1]["chineseName"] = None
    records[1]["isVisible"] = False
    before = deepcopy(records)
    notes["needsReview"]["园艺名"] = "previously unresolved"
    notes["reviewedMappings"] = {
        "园艺名": {
            "status": "linked", "speciesId": "b", "scientificName": "New species",
            "chineseName": "来源中文名", "sources": ["captured-reference"], "note": "已核对",
        }
    }
    result = merge(base, ["园艺名"], notes, records)
    assert result["classes"][-1]["chineseName"] == "来源中文名"
    assert result["classes"][-1]["catalogLocalizationRequired"] is True
    assert result["summary"]["catalogLocalizationRequiredSpecies"] == 1
    assert records == before
    assert result["databaseWritten"] is False


def test_reviewed_identity_requires_exact_database_name_and_evidence():
    base, records, notes = fixtures()
    review = {
        "status": "linked", "speciesId": "b", "scientificName": "Different species",
        "chineseName": "来源中文名", "sources": ["captured-reference"], "note": "已核对",
    }
    notes["reviewedMappings"] = {"园艺名": review}
    with pytest.raises(ValueError, match="identity differs"):
        merge(base, ["园艺名"], notes, records)
    review["scientificName"] = "New species"
    review["sources"] = []
    with pytest.raises(ValueError, match="sources and a reason"):
        merge(base, ["园艺名"], notes, records)


def test_reviewed_ambiguity_preserves_all_candidates_without_linking():
    base, records, notes = fixtures()
    notes["reviewedMappings"] = {
        "新名": {
            "status": "needs-selection", "candidateSpeciesIds": ["a", "b"],
            "sources": ["captured-reference"], "note": "来源列出两个不同物种",
        }
    }
    result = merge(base, ["新名"], notes, records)
    row = result["additionalInputMappings"][0]
    assert row["speciesId"] is None
    assert len(row["candidates"]) == 2
    assert result["summary"]["newDistinctSpecies"] == 0
