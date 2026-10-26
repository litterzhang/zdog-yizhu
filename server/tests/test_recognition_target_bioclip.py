import pytest

from yizhu_server.recognition_target_bioclip import ranked_candidates


def test_ranked_candidates_returns_stable_softmax_order():
    torch = pytest.importorskip("torch")
    classes = [
        {"id": "one", "scientificName": "Example one"},
        {"id": "two", "scientificName": "Example two"},
    ]
    rows = ranked_candidates(torch.tensor([1.0, 2.0]), classes, limit=2)
    assert [row["species_id"] for row in rows] == ["two", "one"]
    assert sum(row["score"] for row in rows) == pytest.approx(1)


def test_ranked_candidates_rejects_nonfinite_logits():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        ranked_candidates(
            torch.tensor([float("nan")]),
            [{"id": "one", "scientificName": "Example one"}],
        )


class FakePresenceClassifier:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def predict(self, image):
        self.calls.append(image.size)
        score = next(self.scores)
        return {"score": score, "containsPlant": False, "modelVersion": "test"}


def test_presence_center_crop_rescues_portrait_plant_without_lowering_threshold():
    image_module = pytest.importorskip("PIL.Image")
    from yizhu_server.recognition_target_bioclip import presence_with_center_fallback

    classifier = FakePresenceClassifier([0.04, 0.32])
    result = presence_with_center_fallback(
        classifier, image_module.new("RGB", (900, 1600)), 0.076,
    )

    assert classifier.calls == [(900, 1600), (900, 900)]
    assert result["containsPlant"] is True
    assert result["selectedView"] == "center_square"
    assert result["viewScores"] == {"whole": 0.04, "center_square": 0.32}


def test_presence_center_crop_is_not_run_after_whole_image_passes():
    image_module = pytest.importorskip("PIL.Image")
    from yizhu_server.recognition_target_bioclip import presence_with_center_fallback

    classifier = FakePresenceClassifier([0.2])
    result = presence_with_center_fallback(
        classifier, image_module.new("RGB", (900, 1600)), 0.076,
    )

    assert classifier.calls == [(900, 1600)]
    assert result["selectedView"] == "whole"
    assert result["viewScores"] == {"whole": 0.2}


def test_presence_center_crop_is_not_run_for_nearly_square_photo():
    image_module = pytest.importorskip("PIL.Image")
    from yizhu_server.recognition_target_bioclip import presence_with_center_fallback

    classifier = FakePresenceClassifier([0.04])
    result = presence_with_center_fallback(
        classifier, image_module.new("RGB", (900, 1000)), 0.076,
    )

    assert classifier.calls == [(900, 1000)]
    assert result["containsPlant"] is False
