import inspect

from yizhu_server.commons_species_search import POLICY, collect


def test_search_policy_is_explicitly_species_only():
    assert POLICY == "commons-explicit-species-search-v1"


def test_deep_search_remains_explicitly_bounded():
    assert "max_pages=3" in str(inspect.signature(collect))
