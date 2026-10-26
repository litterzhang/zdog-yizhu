from yizhu_server.seal_independent_manifest import seal


def test_sealer_is_callable():
    assert callable(seal)
