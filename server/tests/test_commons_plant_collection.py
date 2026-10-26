import json
from copy import deepcopy

import pytest

from yizhu_server.commons_plant_collection import inspect_page, photo_license

SPECIES = {"speciesId": "example:plant", "scientificName": "Osmanthus fragrans", "chineseName": "木犀"}


def page():
    return {
        "pageid": 123, "title": "File:Osmanthus fragrans flowers.jpg",
        "imageinfo": [{
            "thumburl": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/b/image.jpg",
            "thumbmime": "image/jpeg", "sha1": "a" * 40,
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "extmetadata": {
                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                "Artist": {"value": '<a href="/wiki/User:Example">Example photographer</a>'},
                "ImageDescription": {"value": "<i>Osmanthus fragrans</i> in a garden"},
            },
        }],
    }


@pytest.mark.parametrize("license", [
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/licenses/by-nc/4.0/",
    "https://creativecommons.org/licenses/by-nd/4.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
    "https://creativecommons.org.evil.example/licenses/by/4.0/",
    "",
])
def test_only_explicit_per_image_allowlisted_licenses(license):
    assert photo_license({"LicenseUrl": {"value": license}}) is None


def test_by_sa_requires_explicit_independent_collection_opt_in():
    metadata = {
        "LicenseUrl": {
            "value": "https://creativecommons.org/licenses/by-sa/4.0/",
        },
    }
    assert photo_license(metadata, allow_by_sa=True) == (
        "cc-by-sa",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    )


def test_cc0_and_ccby_are_not_confused_with_generic_public_domain():
    assert photo_license({"LicenseUrl": {"value": "http://creativecommons.org/licenses/by/2.0"}})[0] == "cc-by"
    assert photo_license({"LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0/"}})[0] == "cc0"
    assert photo_license({"LicenseShortName": {"value": "Public domain"}}) is None


def test_canonicalizes_regional_and_deed_license_urls():
    assert photo_license({
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/3.0/us/deed.en"},
    }) == ("cc-by", "https://creativecommons.org/licenses/by/3.0/")
    assert photo_license({
        "LicenseUrl": {"value": "http://creativecommons.org/publicdomain/zero/1.0/deed.en"},
    }) == ("cc0", "https://creativecommons.org/publicdomain/zero/1.0/")


def test_category_candidate_is_not_automatically_training_ground_truth():
    original = page()
    snapshot = deepcopy(original)
    row, reason = inspect_page(original, SPECIES)
    assert reason is None and row["speciesId"] == SPECIES["speciesId"]
    assert row["license"] == "cc-by" and row["attribution"] == "Example photographer"
    assert row["trainingEligible"] is False and row["split"] == "unassigned"
    assert row["labelReviewStatus"] == "metadata-only"
    assert original == snapshot


def test_original_mime_is_used_when_api_omits_identical_thumbnail_mime():
    row = page()
    del row["imageinfo"][0]["thumbmime"]
    row["imageinfo"][0]["mime"] = "image/jpeg"
    assert inspect_page(row, SPECIES)[1] is None
    row["imageinfo"][0]["mime"] = "image/svg+xml"
    assert inspect_page(row, SPECIES)[1] == "unsupported-image-location-or-format"


@pytest.mark.parametrize("text", [
    "Osmanthus fragrans distribution map", "Osmanthus fragrans dried flowers",
    "Osmanthus fragrans coloured etching", "Osmanthus fragrans herbarium specimen",
])
def test_non_photo_or_processed_plant_metadata_not_collected(text):
    row = page()
    row["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = text
    assert inspect_page(row, SPECIES)[1] == "not-a-live-plant-photo"


def test_generic_category_membership_without_scientific_name_is_not_enough():
    row = page()
    row["title"] = "File:Garden.jpg"
    row["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = "A garden"
    assert inspect_page(row, SPECIES)[1] == "no-explicit-scientific-name"


def test_hybrid_multiplication_sign_matches_plain_x_metadata():
    row = page()
    species = {
        **SPECIES, "scientificName": "Fatshedera lizei",
        "commonsScientificName": "× Fatshedera lizei",
    }
    row["title"] = "File:X Fatshedera lizei leaves.jpg"
    row["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = "x Fatshedera lizei"
    assert inspect_page(row, species, allow_by_sa=True)[1] is None


def test_exact_species_category_can_use_explicit_target_chinese_name():
    row = page()
    row["title"] = "File:早园竹.jpg"
    row["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = "植物早园竹"
    species = {
        "speciesId": "example:bamboo", "scientificName": "Phyllostachys propinqua",
        "chineseName": "早园竹",
    }
    candidate, reason = inspect_page(row, species)
    assert reason is None
    assert candidate["sourceNameMatch"] == "target-chinese-name"


def test_reviewed_traditional_chinese_source_alias_is_supported():
    row = page()
    row["title"] = "File:ZouJyunZuk.jpg"
    row["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = "公園入面種嘅早園竹"
    species = {
        "speciesId": "example:bamboo", "scientificName": "Phyllostachys propinqua",
        "chineseName": "早园竹", "commonsChineseNames": ["早園竹"],
    }
    assert inspect_page(row, species)[1] is None


@pytest.mark.parametrize("url", [
    "http://upload.wikimedia.org/wikipedia/commons/x.jpg",
    "https://localhost/wikipedia/commons/x.jpg",
    "https://upload.wikimedia.org.evil.example/wikipedia/commons/x.jpg",
    "https://user:pass@upload.wikimedia.org/wikipedia/commons/x.jpg",
    "https://upload.wikimedia.org:9000/wikipedia/commons/x.jpg",
])
def test_download_host_and_scheme_allowlist(url):
    row = page()
    row["imageinfo"][0]["thumburl"] = url
    assert inspect_page(row, SPECIES)[1] == "unsupported-image-location-or-format"


def test_missing_attribution_and_additional_restrictions_fail_closed():
    row = page()
    row["imageinfo"][0]["extmetadata"]["Artist"]["value"] = ""
    assert inspect_page(row, SPECIES)[1] == "missing-attribution"
    row = page()
    row["imageinfo"][0]["extmetadata"]["Restrictions"] = {"value": "restrictions"}
    assert inspect_page(row, SPECIES)[1] == "additional-restrictions"


def test_source_restriction_stops_without_retry_and_preserves_counts(tmp_path, monkeypatch):
    import sys

    from yizhu_server import commons_plant_collection as collector
    from yizhu_server.training_data import write_json

    write_json(tmp_path / "progress.json", {"downloadedCandidateImages": 18, "stage": "collecting"})
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        raise PermissionError("Public source paused: HTTP 429")

    monkeypatch.setattr(collector, "collect", blocked)
    monkeypatch.setattr(sys, "argv", [
        "collector", "--targets", "targets.json", "--primary", "primary",
        "--output", str(tmp_path),
    ])
    with pytest.raises(PermissionError):
        collector.main()
    assert calls == [1]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "stopped-source-restriction"
    assert progress["downloadedCandidateImages"] == 18
    assert progress["automaticRetry"] is False
