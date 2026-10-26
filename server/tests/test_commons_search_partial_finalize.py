from yizhu_server.commons_search_partial_finalize import finalize


def test_partial_finalizer_is_callable():
    assert callable(finalize)
