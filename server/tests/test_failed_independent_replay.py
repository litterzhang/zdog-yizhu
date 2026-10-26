from yizhu_server.failed_independent_replay import create_plan


def test_replay_planner_is_callable():
    assert callable(create_plan)
