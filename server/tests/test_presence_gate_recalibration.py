from yizhu_server.presence_gate_recalibration import recalibrate


def test_recalibration_is_callable():
    assert callable(recalibrate)
