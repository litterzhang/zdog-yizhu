from yizhu_server.independent_exclusions import build


def test_builder_is_callable():
    assert callable(build)
