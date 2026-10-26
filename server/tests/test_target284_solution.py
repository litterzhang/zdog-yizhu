from yizhu_server.recognition_solutions import get_solution


def test_target284_succulent_solution_is_registered_and_production():
    solution = get_solution("target284-succulent-bioclip2-replay284")
    assert solution.experimental is False
    assert solution.model_directory == "target284-succulent-replay284-center-l2-inverse-lowreg-v1"
    assert solution.version == "20261005.2"


def test_target285_paraguayense_solution_is_registered_and_production():
    solution = get_solution("target285-paraguayense-bioclip2")
    assert solution.experimental is False
    assert solution.version == "20261006.1"
    assert solution.model_directory == "target285-paraguayense-frozen-old-v1"
    assert solution.model_sha256 == (
        "ba07fb5ca90d497fead259a35420bcda0c7bc84262edfad8afda3bcdafd80bd1"
    )
