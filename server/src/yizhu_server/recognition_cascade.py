"""Bounded local multiview retrieval and PlantCLEF fallback; no external services."""

import copy
import math
import time

from yizhu_server.recognition_rerank import validate_candidates

POLICY = "cross-view-agreement-v1"


def crop_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
  if min(width, height) < 64:
    return []
  side = max(1, int(min(width, height) * 0.8))
  boxes = []
  for fraction in (1 / 3, 2 / 3):
    cx, cy = (width * fraction, height / 2) if width >= height else (width / 2, height * fraction)
    left = max(0, min(width - side, round(cx - side / 2)))
    top = max(0, min(height - side, round(cy - side / 2)))
    box = (left, top, left + side, top + side)
    if box not in boxes:
      boxes.append(box)
  return boxes


def checked(result: dict) -> None:
  if result.get("status") not in {"requires_review", "uncertain"}:
    raise ValueError("Unexpected retrieval status.")
  validate_candidates(result.get("candidates"))
  if any(not isinstance(c.get("score"), (int, float)) or isinstance(c["score"], bool)
         or not math.isfinite(c["score"]) for c in result["candidates"]):
    raise ValueError("Invalid retrieval similarity.")


def fuse_views(views: list[dict]) -> dict:
  """Keep all original top-50 references, then add at most 50 RRF-ranked new ones."""
  if not 1 <= len(views) <= 3:
    raise ValueError("Use one whole image and at most two crops.")
  for view in views:
    checked(view["result"])
  original = views[0]["result"]
  baseline = original["candidates"]
  if len(baseline) > 50 or any(len(v["result"]["candidates"]) > 50 for v in views):
    raise ValueError("Each recall view is bounded at 50.")
  found = {c["species_id"]: copy.deepcopy(c) for c in baseline}
  extras = {}
  for view in views[1:]:
    if view["result"]["status"] == "uncertain":
      continue
    for rank, candidate in enumerate(view["result"]["candidates"], 1):
      identifier = candidate["species_id"]
      if identifier in found:
        continue
      value = extras.setdefault(identifier, {"candidate": copy.deepcopy(candidate), "rrf": 0.0})
      value["rrf"] += 1 / (60 + rank)
  for identifier, value in sorted(extras.items(), key=lambda pair: (-pair[1]["rrf"], pair[0]))[:50]:
    found[identifier] = {**value["candidate"], "recall_rrf_score": value["rrf"]}
  return {
    **copy.deepcopy(original), "candidates": list(found.values()),
    "scope": "multiview_reference_shortlist",
    "multiview": {
      "version": "1", "views": copy.deepcopy(views), "rrf_k": 60,
      "protected_count": len(baseline), "candidate_count": len(found),
    },
  }


def support(identifier: str, views: list[dict]) -> dict:
  top_five, first = [], []
  for view in views:
    result = view["result"]
    if result["status"] == "uncertain":
      continue
    ids = [row["species_id"] for row in result["candidates"]]
    if identifier in ids[:5]:
      top_five.append(view["id"])
    if ids and ids[0] == identifier:
      first.append(view["id"])
  return {"top_five_views": top_five, "first_views": first}


def unique_positive_lead(candidates: list[dict]) -> bool:
  scores = [c["score"] for c in candidates[:2]]
  return bool(scores and all(isinstance(s, (int, float)) and not isinstance(s, bool)
                             and math.isfinite(s) for s in scores)
              and scores[0] > 0 and (len(scores) == 1 or scores[0] > scores[1] + 1e-8))


def interleave(primary: list[dict], fallback: list[dict]) -> list[dict]:
  """Suggestion order, not a numerical comparison of cosine and softmax scores."""
  rows, seen = [], set()
  for index in range(max(len(primary), len(fallback))):
    for candidates in (primary, fallback):
      if index >= len(candidates):
        continue
      row = candidates[index]
      identifier = row.get("species_id")
      if row.get("kind") != "plant" or not identifier or identifier in seen:
        continue
      seen.add(identifier)
      rows.append(copy.deepcopy(row))
  return rows


class CascadeProvider:
  def __init__(self, solution, retrieval, reranker, fallback, catalog):
    self.solution = solution
    self.retrieval = retrieval
    self.reranker = reranker
    self.fallback = fallback
    self.catalog = catalog

  def recognize(self, image, *, top_k=5):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
      raise ValueError("top_k must be between 1 and 20.")
    started = time.perf_counter()
    before = time.perf_counter()
    base = self.retrieval.retrieve(image, limit=50)
    timings = {"retrieval_seconds": time.perf_counter() - before, "rerank_seconds": 0.0,
               "plantclef_seconds": 0.0}
    checked(base)
    views = [{"id": "whole", "box": [0, 0, *image.size], "result": base}]
    # Infrastructure exceptions propagate; a technical failure is not uncertain taxonomy.
    if base["status"] == "uncertain":
      return self._pack(
        base, None, base["candidates"], "uncertain", base.get("reason"),
        {"accepted": False, "reason": "whole_image_abstained"},
        {"called": False, "reason": "whole_image_abstained"}, top_k, started, timings,
      )
    for index, box in enumerate(crop_boxes(*image.size), 1):
      before = time.perf_counter()
      with image.crop(box) as crop:
        result = self.retrieval.retrieve(crop, limit=50)
      timings["retrieval_seconds"] += time.perf_counter() - before
      checked(result)
      views.append({"id": f"crop-{index}", "box": list(box), "result": result})
    recalled = fuse_views(views)
    before = time.perf_counter()
    primary = self.reranker.rerank(image, recalled, view="official")
    timings["rerank_seconds"] = time.perf_counter() - before
    checked(primary)
    # A reranker may only reorder the provided pool, never invent IDs.
    expected = {c["species_id"] for c in recalled["candidates"]}
    if {c["species_id"] for c in primary["candidates"]} != expected:
      raise ValueError("Reranker changed recall pool membership.")
    leader = primary["candidates"][0]["species_id"]
    agreement = support(leader, views)
    visible_leader = self.catalog.get(leader) is not None
    accepted = (
      primary["status"] != "uncertain" and visible_leader
      and unique_positive_lead(primary["candidates"])
      and len(agreement["top_five_views"]) >= 2 and bool(agreement["first_views"])
    )
    primary_gate = {"accepted": accepted, "visible_leader": visible_leader, **agreement}
    if accepted:
      return self._pack(
        primary, None, primary["candidates"], "requires_review", None,
        primary_gate, {"called": False, "reason": "primary_agreement"}, top_k, started, timings,
      )

    before = time.perf_counter()
    secondary = self.fallback.recognize(image, top_k=20)
    timings["plantclef_seconds"] = time.perf_counter() - before
    candidates = secondary.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20:
      raise ValueError("PlantCLEF must return bounded raw class evidence.")
    for row in candidates:
      if (row.get("kind") not in {"plant", "unmapped"}
          or (row["kind"] == "plant" and not isinstance(row.get("species_id"), str))
          or isinstance(row.get("score"), bool) or not isinstance(row.get("score"), (int, float))
          or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
        raise ValueError("Invalid PlantCLEF class evidence.")
    best = secondary["candidates"][0]
    identifier = best.get("species_id")
    fallback_support = support(identifier, views) if identifier else {
      "top_five_views": [], "first_views": [],
    }
    # Without a calibrated open-set detector, a closed-set classifier cannot
    # confidently introduce an unseen taxon solely on its softmax value.
    # New secondary candidates are still available for manual confirmation below.
    accepted = (
      visible_leader and primary["status"] != "uncertain"
      and best.get("kind") == "plant" and identifier is not None
      and self.catalog.get(identifier) is not None
      and unique_positive_lead(secondary["candidates"])
      and identifier in [row["species_id"] for row in primary["candidates"][:5]]
      and len(fallback_support["top_five_views"]) >= 2
      and bool(fallback_support["first_views"])
    )
    fallback_gate = {
      "called": True, "accepted": bool(accepted), **fallback_support,
      "reason": "cross_model_agreement" if accepted else "no_supported_agreement",
    }
    if accepted:
      rows = secondary["candidates"]
      status, reason = "requires_review", None
    else:
      rows = interleave(primary["candidates"], secondary["candidates"])
      status = "uncertain"
      reason = "hidden_leading_candidate" if not visible_leader else "cascade_no_agreement"
    return self._pack(
      primary, secondary, rows, status, reason, primary_gate, fallback_gate, top_k, started, timings,
    )

  def _pack(self, primary, secondary, rows, status, reason, primary_gate, fallback_gate, top_k,
            started, timings):
    candidates = copy.deepcopy(rows[:top_k])
    for row in candidates:
      row["confidence"] = None
    return {
      "source": "BioCLIP + PlantCLEF" if secondary is not None else "BioCLIP",
      "model": self.solution.id, "revision": f"{self.solution.id}-{self.solution.version}",
      "provider_version": f"{self.solution.id}-{self.solution.version}",
      "is_demo": False, "status": status, "reason": reason,
      "score_type": "stage_scores_not_comparable_or_calibrated" if secondary is not None else
                    "cosine_similarity_not_calibrated_confidence",
      "scope": "bounded_local_cascade", "label_version": primary.get("label_version"),
      "catalog_plant_count": primary.get("catalog_plant_count"),
      "solution": self.solution.public_info(), "candidates": candidates,
      "cascade": {
        "policy": POLICY, "calibrated": False,
        "primary_gate": primary_gate, "fallback_gate": fallback_gate,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "timings": {key: round(value, 4) for key, value in timings.items()},
        "primary": copy.deepcopy(primary), "plantclef": copy.deepcopy(secondary),
      },
    }
