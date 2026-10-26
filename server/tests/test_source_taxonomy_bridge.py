from copy import deepcopy

import pytest

from yizhu_server.source_taxonomy_bridge import accepted_key, build_bridge, validate_bridge
from yizhu_server.training_data import PHOTO_HOST, public_samples


def fixtures():
    target = {"speciesId": "example:hamabo", "scientificName": "Talipariti hamabo"}
    source = {
        "id": 1106438, "name": "Hibiscus hamabo", "rank": "species", "is_active": True,
        "ancestor_ids": [48460, 47126],
    }
    target_match = {
        "canonicalName": "Talipariti hamabo", "usageKey": 3937028, "status": "ACCEPTED",
        "rank": "SPECIES", "matchType": "EXACT",
    }
    source_match = {
        "canonicalName": "Hibiscus hamabo", "usageKey": 7279115, "acceptedUsageKey": 3937028,
        "status": "SYNONYM", "rank": "SPECIES", "matchType": "EXACT",
    }
    return target, source, target_match, source_match


def test_bridge_preserves_catalog_identity_and_real_external_evidence():
    args = fixtures()
    before = deepcopy(args)
    result = build_bridge(*args)
    validate_bridge(result)
    assert result["speciesId"] == args[0]["speciesId"]
    assert result["scientificName"] == "Talipariti hamabo"
    assert result["sourceScientificName"] == "Hibiscus hamabo"
    assert "linkedScientificNames" not in result["sourceNameEvidence"]
    assert args == before


@pytest.mark.parametrize("change", [
    {"matchType": "FUZZY"}, {"rank": "GENUS"}, {"status": "DOUBTFUL"},
    {"acceptedUsageKey": 999}, {"canonicalName": "Hibiscus other"},
])
def test_fuzzy_higher_rank_or_different_accepted_taxon_is_rejected(change):
    target, source, tm, sm = fixtures()
    sm.update(change)
    with pytest.raises(ValueError):
        build_bridge(target, source, tm, sm)


def test_source_rank_and_identity_cannot_be_swapped():
    result = build_bridge(*fixtures())
    for field, value in [("id", 999), ("rank", "hybrid"), ("is_active", False)]:
        changed = deepcopy(result)
        changed["sourceNameEvidence"]["inatTaxon"][field] = value
        with pytest.raises(ValueError):
            validate_bridge(changed)


def test_public_samples_still_require_exact_source_taxon_and_license():
    result = build_bridge(*fixtures())
    observation = {
        "id": 1, "quality_grade": "research", "user": {"id": 2},
        "taxon": {"id": 1106438, "name": "Hibiscus hamabo", "rank": "species"},
        "photos": [{"id": 3, "license_code": "cc-by", "attribution": "Synthetic photographer",
                    "url": f"https://{PHOTO_HOST}/photos/3/square.jpg"}],
    }
    rows = public_samples([observation], result)
    assert rows[0]["scientificName"] == "Talipariti hamabo"
    assert rows[0]["sourceNameEvidence"]["matchType"] == "gbif-exact-synonym"
    observation["taxon"]["id"] = 999
    assert public_samples([observation], result) == []


def test_unknown_accepted_identity_is_not_inferred():
    with pytest.raises(ValueError):
        accepted_key({"canonicalName": "Example name", "rank": "SPECIES",
                      "matchType": "EXACT", "usageKey": 1, "status": "SYNONYM"}, "Example name")
