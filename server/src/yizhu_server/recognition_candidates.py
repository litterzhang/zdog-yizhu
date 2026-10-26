"""Candidate display policy only. Never changes model scores or confirmation."""

import math

POLICY = "top-score-ratio-v1"
MINIMUM_TOP_SCORE_RATIO = 0.5


def policy_info():
  return {
    "policy": POLICY, "minimumTopScoreRatio": MINIMUM_TOP_SCORE_RATIO,
    "maximumCandidates": 5, "confidenceCalibrated": False,
  }


def close_candidates(candidates):
  """Keep the leading score's close competitors; preserve order, ties and raw values."""
  if not isinstance(candidates, list) or any(not isinstance(row, dict) for row in candidates):
    raise ValueError("Candidates must be a list of score records.")
  rows = list(candidates)
  scores = [row.get("score") for row in rows]
  if any(
    not isinstance(score, (int, float)) or isinstance(score, bool)
    or not math.isfinite(score) or not 0 <= score <= 1
    for score in scores
  ) or any(a < b for a, b in zip(scores, scores[1:])):
    raise ValueError("Candidate display requires finite, descending softmax scores.")
  if not rows:
    return []
  cutoff = scores[0] * MINIMUM_TOP_SCORE_RATIO
  return [row for row in rows[:5] if row["score"] >= cutoff]


def present_recognition(record):
  """Project historical rows without rewriting persisted recognition/confirmation data."""
  raw = record.get("rawResult")
  if (
    record.get("status") != "succeeded" or not isinstance(raw, dict)
    or raw.get("source") != "Pl@ntNet"
    or raw.get("scoreType") != "softmax_not_calibrated_confidence"
    or raw.get("reason")
  ):
    return record
  candidates = raw.get("candidates", [])
  try:
    selected = close_candidates(candidates)
  except ValueError:
    return record  # A malformed legacy score must not break the whole private journal.
  return {
    **record,
    "rawResult": {
      **raw, "candidates": selected,
      "candidateSelection": {**policy_info(), "returnedCount": len(selected)},
    },
  }
