import inspect

from yizhu_server.target_model_metadata import build


def test_target_metadata_accepts_explicit_presence_gate_contract():
    assert "presence_gate_path" in inspect.signature(build).parameters
