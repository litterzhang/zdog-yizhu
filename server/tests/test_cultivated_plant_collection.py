from copy import deepcopy

import pytest

from yizhu_server.cultivated_plant_collection import consensus_candidate
from yizhu_server.training_data import PHOTO_HOST, public_samples

SPECIES = {
    "speciesId": "example:osmanthus", "scientificName": "Osmanthus fragrans", "inatTaxonId": 371103,
}


def taxon(identifier=371103, rank="species"):
    return {"id": identifier, "rank": rank, "name": "Osmanthus fragrans",
            "ancestor_ids": [47126, 371103, identifier]}


def observation():
    return {
        "id": 100, "user": {"id": 200}, "quality_grade": "casual", "captive": True,
        "identifications_most_agree": True, "identifications_most_disagree": False,
        "community_taxon_id": 371103, "observed_on": "2025-10-01",
        "location": "private-location-must-not-be-saved",
        "quality_metrics": [{"metric": "wild", "agree": False, "user": {"secret-profile": "x"}}],
        "taxon": taxon(),
        "identifications": [
            {"id": 1, "current": True, "category": "improving", "disagreement": None,
             "user": {"id": 200}, "taxon": taxon()},
            {"id": 2, "current": True, "category": "supporting", "disagreement": False,
             "user": {"id": 201}, "taxon": taxon()},
        ],
        "photos": [
            {"id": 300, "license_code": "cc-by", "attribution": "Synthetic photographer",
             "url": f"https://{PHOTO_HOST}/photos/300/square.jpg"},
        ],
    }


def test_real_grade_preserved_and_candidate_not_auto_admitted_or_location_saved():
    original = observation()
    before = deepcopy(original)
    candidate, reason = consensus_candidate(original, SPECIES)
    assert reason is None
    assert candidate["sourceQualityGrade"] == "casual"
    assert candidate["trainingEligible"] is False and "split" not in candidate
    assert candidate["proposedSplit"] in {"train", "validation", "test"}
    assert "location" not in candidate and "user" not in candidate
    assert candidate["consensusEvidence"]["qualityMetrics"] == [{"metric": "wild", "agree": False}]
    assert len(candidate["consensusEvidence"]["agreeingAccounts"]) == 2
    assert original == before
    # Main research-grade collector remains strict; no grade is rewritten.
    assert public_samples([original], SPECIES) == []


@pytest.mark.parametrize("change", [
    {"captive": False}, {"identifications_most_agree": False},
    {"identifications_most_disagree": True}, {"community_taxon_id": 47126},
    {"community_taxon_id": None}, {"observed_on": None}, {"location": None},
    {"quality_metrics": [{"metric": "evidence", "agree": False}]},
])
def test_missing_consensus_or_other_quality_failures_rejected(change):
    assert consensus_candidate({**observation(), **change}, SPECIES)[0] is None


def test_at_least_two_distinct_current_agreeing_accounts_are_required():
    for change in ["one", "same-account", "old-identification", "conflict", "disagreement"]:
        row = observation()
        if change == "one":
            row["identifications"].pop()
        elif change == "same-account":
            row["identifications"][1]["user"]["id"] = 200
        elif change == "old-identification":
            row["identifications"][1]["current"] = False
        elif change == "conflict":
            row["identifications"][1]["taxon"] = {"id": 999, "rank": "species", "ancestor_ids": [47126]}
        else:
            row["identifications"][1]["disagreement"] = True
        assert consensus_candidate(row, SPECIES)[0] is None


def test_variety_maps_to_species_only_with_explicit_ancestor_link():
    row = observation()
    row["taxon"] = taxon(566373, "variety")
    row["community_taxon_id"] = 566373
    for vote in row["identifications"]:
        vote["taxon"] = taxon(566373, "variety")
    candidate, reason = consensus_candidate(row, SPECIES)
    assert reason is None and candidate["speciesId"] == SPECIES["speciesId"]
    assert candidate["scientificName"] == SPECIES["scientificName"]
    row["taxon"]["ancestor_ids"] = [47126]
    assert consensus_candidate(row, SPECIES)[0] is None


@pytest.mark.parametrize("license", ["cc-by-nc", "cc-by-sa", None])
def test_photo_license_not_relaxed_for_cultivated_records(license):
    row = observation()
    row["photos"][0]["license_code"] = license
    assert consensus_candidate(row, SPECIES)[0] is None
