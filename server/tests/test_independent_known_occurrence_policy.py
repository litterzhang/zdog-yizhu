import inspect

from yizhu_server.target162_independent_test import audit


def test_known_occurrence_relaxation_is_explicit_opt_in():
    parameter = inspect.signature(audit).parameters["allow_known_occurrences"]
    assert parameter.default is False
