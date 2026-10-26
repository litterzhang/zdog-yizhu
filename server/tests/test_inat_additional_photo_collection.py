from yizhu_server.inat_additional_photo_collection import LICENSES


def test_only_explicit_creative_commons_photo_licenses_are_allowed():
    assert LICENSES == {"cc0", "cc-by", "cc-by-sa"}
