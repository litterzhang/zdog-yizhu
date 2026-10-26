from yizhu_server.gbif_additional_media_collection import eligible_media


def test_additional_media_collector_is_explicit():
    assert callable(eligible_media)
