import inspect

from yizhu_server.target_only_feature_training import train


def test_training_accepts_prior_target_classifier_initialization():
    parameters = inspect.signature(train).parameters
    assert "target_initialization" in parameters
    assert "freeze_target_initialized" in parameters
