from yizhu_server.gbif_target_collection import BY_SA_LICENSE, LICENSES


def test_by_sa_is_checked_at_media_level_not_invalid_gbif_filter():
    assert BY_SA_LICENSE not in LICENSES
