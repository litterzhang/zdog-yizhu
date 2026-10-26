"""Closed output contract for completed, clean-start, locally trained species models."""

import copy
import hashlib
import json
import re


def _digest(value):
  payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode()).hexdigest()


def build_trained_contract(
  classes, *, target_species_ids, training_examples, completed_epochs,
  backbone_optimizer_steps, checkpoint_sha256, initialization="random",
):
  """Call only after training; counts must reflect consumed, labeled training images."""
  if (
    initialization != "random"
    or type(completed_epochs) is not int or completed_epochs < 1
    or type(backbone_optimizer_steps) is not int or backbone_optimizer_steps < 1
    or not isinstance(checkpoint_sha256, str)
    or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
  ):
    raise ValueError("Require completed clean-start backbone training and a pinned checkpoint.")
  if not isinstance(classes, list) or not 2 <= len(classes) <= 4096:
    raise ValueError("Require a finite trained class list.")
  if (
    not isinstance(target_species_ids, list)
    or any(not isinstance(sid, str) or not sid for sid in target_species_ids)
    or len(set(target_species_ids)) != len(target_species_ids)
  ):
    raise ValueError("Require unique approved target species IDs.")
  allowed = set(target_species_ids)
  ids, scientific_names, trained = [], [], []
  if not isinstance(training_examples, dict):
    raise ValueError("Require actual per-class training-image counts.")
  for row in classes:
    sid, scientific = row.get("speciesId"), row.get("scientificName")
    count = training_examples.get(sid)
    if (
      sid not in allowed or not isinstance(scientific, str) or not scientific.strip()
      or row.get("classRole") not in {None, "trained-from-scratch"}
      or row.get("sourceClasses")
      or type(count) is not int or count < 1
    ):
      raise ValueError("Output class is untrained, inherited, or outside the approved targets.")
    ids.append(sid)
    scientific_names.append(scientific)
    trained.append({
      "speciesId": sid, "scientificName": scientific,
      "chineseName": row.get("chineseName"), "classRole": "trained-from-scratch",
      "trainingExamples": count,
    })
  if (
    len(set(ids)) != len(ids) or len(set(scientific_names)) != len(scientific_names)
    or set(training_examples) != set(ids)
  ):
    raise ValueError("Training labels and output classes must agree exactly.")
  contract = {
    "schemaVersion": 1, "task": "species-classification", "status": "trained",
    "initialization": "random", "checkpointSha256": checkpoint_sha256,
    "classCount": len(trained), "classes": trained,
    "targetSpeciesIds": list(target_species_ids),
    "completedEpochs": completed_epochs,
    "backboneOptimizerSteps": backbone_optimizer_steps,
    "inheritedClassCount": 0, "pretrainedWeightsUsed": False,
    "pretrainedFeaturesUsed": False, "externalSpeciesFallbackAllowed": False,
    "confidenceCalibrated": False, "productionApproved": False,
  }
  contract["contractSha256"] = _digest(contract)
  return contract


def validate_trained_contract(contract):
  """Reject stale/expanded labels rather than silently inheriting a global catalog."""
  if not isinstance(contract, dict):
    raise ValueError("Invalid trained output contract.")
  try:
    supplied = copy.deepcopy(contract)
    checksum = supplied.pop("contractSha256")
    if checksum != _digest(supplied):
      raise ValueError("Trained output contract changed.")
    rebuilt = build_trained_contract(
      supplied["classes"], target_species_ids=supplied["targetSpeciesIds"],
      training_examples={
        row["speciesId"]: row["trainingExamples"] for row in supplied["classes"]
      },
      completed_epochs=supplied["completedEpochs"],
      backbone_optimizer_steps=supplied["backboneOptimizerSteps"],
      checkpoint_sha256=supplied["checkpointSha256"],
      initialization=supplied["initialization"],
    )
    if rebuilt != contract:
      raise ValueError("Trained output policy differs from the clean-model policy.")
  except (KeyError, TypeError, AttributeError) as exc:
    raise ValueError("Incomplete trained output contract.") from exc
  return contract


def rank_trained_candidates(logits, contract, *, top_k=5):
  """Return only trained row IDs; never query a pretrained label set or fill from a catalog."""
  import torch

  validate_trained_contract(contract)
  classes = contract["classes"]
  if (
    type(top_k) is not int or not 1 <= top_k <= 5
    or not isinstance(logits, torch.Tensor)
    or logits.shape != (len(classes),) or not logits.is_floating_point()
    or not torch.isfinite(logits).all()
  ):
    raise ValueError("Logits must have exactly one value per trained species.")
  with torch.inference_mode():
    order = torch.argsort(logits, descending=True, stable=True)[:min(top_k, len(classes))]
    scores = logits.softmax(0)
    return [{
      "speciesId": classes[i]["speciesId"],
      "scientificName": classes[i]["scientificName"],
      "chineseName": classes[i]["chineseName"],
      "score": float(scores[i]), "confidence": None,
    } for i in order.tolist()]
