"""Adapt the pinned two-stage Pl@ntNet model to the existing private-journal API."""

import math
from pathlib import Path

from yizhu_server.plantnet_transfer import INCREMENTAL_ARCHITECTURE, PlantNetCascade
from yizhu_server.recognition_candidates import close_candidates, policy_info
from yizhu_server.training_data import sha256

SERVER = Path(__file__).resolve().parents[2]


class PlantNetRecognitionProvider:
  def __init__(self, solution, catalog, *, threads=2):
    if not catalog.available:
      raise ValueError("Install the Chinese catalog before starting recognition.")
    if not solution.model_directory or not solution.model_sha256:
      raise ValueError("Select a pinned trained model, not a planned class list.")
    directory = (SERVER / ".artifacts/models" / solution.model_directory).resolve()
    if not directory.is_relative_to((SERVER / ".artifacts/models").resolve()):
      raise ValueError("Model directory escapes the local artifact store.")
    if sha256(directory / "model.json") != solution.model_sha256:
      raise ValueError("Registered model metadata checksum mismatch.")
    self.pipeline = PlantNetCascade(directory, threads=threads)
    if (
      self.pipeline.model["architecture"] != INCREMENTAL_ARCHITECTURE
      or self.pipeline.version != solution.model_sha256
    ):
      raise ValueError("The selected model is not the verified incremental cascade.")
    self.solution, self.catalog = solution, catalog
    self.class_ids = {row["speciesId"] for row in self.pipeline.model["classes"]}
    if len(self.class_ids) != self.pipeline.model["classCount"]:
      raise ValueError("Duplicate trained output IDs.")

  def public_model_info(self):
    model = self.pipeline.model
    return {
      "modelVersion": self.pipeline.version, "supportedPlantCount": model["classCount"],
      "retainedPlantCount": model["retainedClassCount"],
      "trainedAdditionalPlantCount": model["trainedAdditionalClassCount"],
      "plantPresenceGate": True,
      "candidateSelection": policy_info(),
    }

  def recognize(self, image, *, top_k=5):
    if type(top_k) is not int or not 1 <= top_k <= 20:
      raise ValueError("top_k must be between 1 and 20.")
    raw = self.pipeline.predict(image)  # Binary gate first; species encoder only after passing.
    if raw["status"] not in {"requires_review", "nonplant"}:
      raise ValueError("Unexpected two-stage prediction status.")
    if raw["status"] == "nonplant" and raw["candidates"]:
      raise ValueError("A blocked photo must not produce species candidates.")
    candidates = []
    for row in raw["candidates"][:min(top_k, 5)]:
      if (
        row["speciesId"] not in self.class_ids
        or not isinstance(row["score"], (int, float)) or isinstance(row["score"], bool)
        or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1
      ):
        raise ValueError("Unexpected species output from the trained model.")
      candidates.append({
        "species_id": row["speciesId"], "scientific_name": row["scientificName"],
        "label": row["scientificName"], "kind": "plant",
        "score": row["score"], "confidence": None,
      })
    hidden_leader = bool(candidates and self.catalog.get(candidates[0]["species_id"]) is None)
    reason = (
      "nonplant_gate" if raw["status"] == "nonplant"
      else "hidden_leading_candidate" if hidden_leader
      else "no_trained_candidates" if not candidates else None
    )
    displayed = close_candidates(candidates) if not reason else candidates
    return {
      "source": "Pl@ntNet", "model": INCREMENTAL_ARCHITECTURE,
      "revision": self.pipeline.version, "label_version": self.pipeline.model["speciesSha256"],
      "provider_version": f"plantnet-china-{self.solution.version}",
      "is_demo": False, "status": "uncertain" if reason else "requires_review", "reason": reason,
      "scope": "trained_chinese_species_closed_set",
      "score_type": "softmax_not_calibrated_confidence",
      "catalog_plant_count": len(self.class_ids), "candidates": displayed,
      "candidate_selection": {
        **policy_info(), "applied": reason is None,
        "originalCount": len(raw["candidates"]), "returnedCount": len(displayed),
      },
      "solution": {**self.solution.public_info(), **self.public_model_info()},
      "presence_gate": {
        "score": raw["presenceScore"], "threshold": raw["presenceThreshold"],
        "passed": raw["status"] != "nonplant", "confidence": None,
        "model_version": self.pipeline.model["presenceGate"]["headSha256"],
      },
      "original_prediction": raw,
    }
