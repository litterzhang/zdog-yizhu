from copy import deepcopy

import pytest

from yizhu_server.named_hybrid_sources import (
    binomial_notation,
    resolve_named_hybrid,
    validate_hybrid_evidence,
)
from yizhu_server.training_data import PHOTO_HOST, public_samples


def species():
    return {"speciesId": "example:strawberry", "scientificName": "Fragaria ananassa"}


def payload():
    return {
        "total_results": 1, "results": [{
            "id": 55366, "name": "Fragaria × ananassa", "rank": "hybrid",
            "is_active": True, "ancestor_ids": [48460, 47126],
            "names": [{"name": "Fragaria × ananassa", "lexicon": "scientific-names",
                       "is_valid": True}],
        }],
    }


def observation():
    return {
        "id": 100, "quality_grade": "research", "user": {"id": 200},
        "taxon": {"id": 55366, "name": "Fragaria × ananassa", "rank": "hybrid"},
        "photos": [{
            "id": 300, "license_code": "cc-by", "attribution": "Synthetic public author",
            "url": f"https://{PHOTO_HOST}/photos/300/square.jpg",
        }],
    }


def test_standalone_markers_not_parent_formulas_or_infraspecific_names():
    for name in ["Fragaria × ananassa", "Fragaria x ananassa", "Fragaria ananassa"]:
        assert binomial_notation(name) == "Fragaria ananassa"
    assert binomial_notation("× Fatshedera lizei") == "Fatshedera lizei"
    for name in [
        "Fragaria × ananassa cuneifolia", "Fragaria chiloensis × virginiana",
        "Fragaria", "Fragaria 'Cultivar'", None,
    ]:
        assert binomial_notation(name) is None


def test_named_hybrid_keeps_approved_id_and_original_scientific_name():
    row = resolve_named_hybrid(payload(), species())
    assert row["speciesId"] == species()["speciesId"]
    assert row["scientificName"] == "Fragaria ananassa"
    assert row["sourceScientificName"] == "Fragaria × ananassa"
    validate_hybrid_evidence(row)
    samples = public_samples([observation()], row)
    assert len(samples) == 1 and samples[0]["sourceTaxonRank"] == "hybrid"
    assert samples[0]["scientificName"] == "Fragaria ananassa"
    assert samples[0]["sourceScientificName"] == "Fragaria × ananassa"


@pytest.mark.parametrize("change", [
    {"rank": "infrahybrid"}, {"rank": "subspecies"}, {"rank": "variety"},
    {"rank": "species"}, {"is_active": False}, {"ancestor_ids": [48460]},
])
def test_generic_rank_relaxation_is_not_allowed(change):
    data = payload()
    data["results"][0].update(change)
    with pytest.raises(ValueError, match="unique"):
        resolve_named_hybrid(data, species())


def test_ambiguous_or_truncated_results_fail_closed():
    data = payload()
    data["total_results"] = 2
    with pytest.raises(ValueError, match="truncated"):
        resolve_named_hybrid(data, species())
    data["results"].append({**deepcopy(data["results"][0]), "id": 999})
    with pytest.raises(ValueError, match="unique"):
        resolve_named_hybrid(data, species())


def test_explicit_synonym_needed_when_epithet_differs():
    data = payload()
    data["results"][0]["name"] = "Petunia × atkinsiana"
    target = {"speciesId": "example:petunia", "scientificName": "Petunia hybrida"}
    with pytest.raises(ValueError):
        resolve_named_hybrid(data, target)
    data["results"][0]["names"] = [
        {"name": "Petunia × hybrida", "lexicon": "scientific-names", "is_valid": False}
    ]
    assert resolve_named_hybrid(data, target)["sourceScientificName"] == "Petunia × atkinsiana"


def test_hybrid_observations_need_matching_bridge_rank_and_public_license():
    normal = {**species(), "inatTaxonId": 55366}
    assert public_samples([observation()], normal) == []
    bridged = resolve_named_hybrid(payload(), species())
    for field, value in [
        ("sourceTaxonId", 123), ("queryScientificName", "Other species"),
        ("sourceTaxonRank", "species"), ("linkedScientificNames", []),
    ]:
        changed = deepcopy(bridged)
        changed["sourceNameEvidence"][field] = value
        with pytest.raises(ValueError):
            public_samples([observation()], changed)
    for change in ["noncommercial", "infrahybrid", "casual", "other-id"]:
        row = observation()
        if change == "noncommercial":
            row["photos"][0]["license_code"] = "cc-by-nc"
        elif change == "infrahybrid":
            row["taxon"]["rank"] = "infrahybrid"
        elif change == "other-id":
            row["taxon"]["id"] = 999
        else:
            row["quality_grade"] = "casual"
        assert public_samples([row], bridged) == []
