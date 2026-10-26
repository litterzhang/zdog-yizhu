from pathlib import Path


def test_audit_skips_candidates_outside_a_frozen_reduced_target():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/yizhu_server/target162_independent_test.py"
    ).read_text()
    assert 'rejected["outside-frozen-target"] += 1' in source
