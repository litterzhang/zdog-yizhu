from copy import deepcopy

import pytest

from yizhu_server.recognition_candidates import close_candidates, present_recognition


@pytest.mark.parametrize("scores,expected", [
  ([.981, .0178, .00047, .00026, .00019], 1),
  ([.4, .35, .19, .04, .02], 2),
  ([.4, .2, .2, .1, .1], 3),
  ([.2] * 5, 5),
  ([0.] * 5, 5),
  ([], 0),
])
def test_only_close_scores_returned_without_mutating_inputs(scores, expected):
  rows = [{"speciesId": str(i), "score": score, "confidence": None}
          for i, score in enumerate(scores)]
  before = deepcopy(rows)
  result = close_candidates(rows)
  assert len(result) == expected
  assert result == rows[:expected]
  assert rows == before


@pytest.mark.parametrize("scores", [[.2, .3], [float("nan")], [-.1], [1.1], [True], [None]])
def test_invalid_scores_rejected(scores):
  with pytest.raises(ValueError):
    close_candidates([{"score": score} for score in scores])


def test_history_projection_preserves_originals_and_user_confirmation():
  record = {
    "status": "succeeded", "confirmedSpeciesId": "other-species",
    "rawResult": {
      "source": "Pl@ntNet", "scoreType": "softmax_not_calibrated_confidence",
      "candidates": [{"speciesId": "a", "score": .98}, {"speciesId": "b", "score": .01}],
      "original": {"original_prediction": {"candidates": ["untouched"]}},
    },
  }
  before = deepcopy(record)
  projected = present_recognition(record)
  assert len(projected["rawResult"]["candidates"]) == 1
  assert projected["confirmedSpeciesId"] == "other-species"
  assert projected["rawResult"]["original"] == before["rawResult"]["original"]
  assert record == before
  for changes in (
    {"source": "BioCLIP"}, {"scoreType": "cosine_similarity_not_calibrated_confidence"},
    {"reason": "hidden_leading_candidate"}, {"candidates": [{"score": None}]},
  ):
    value = {**record, "rawResult": {**record["rawResult"], **changes}}
    assert present_recognition(value) == value
  uncertain = {**record, "status": "uncertain"}
  assert present_recognition(uncertain) == uncertain
