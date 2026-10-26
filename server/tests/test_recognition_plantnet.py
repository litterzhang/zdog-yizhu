from copy import deepcopy
from types import SimpleNamespace

import pytest

from yizhu_server import catalog_recognition, recognition_plantnet
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.recognition_solutions import create_provider, get_solution


def pipeline_result(nonplant=False):
  return {
    "status": "nonplant" if nonplant else "requires_review",
    "presenceScore": .02 if nonplant else .8, "presenceThreshold": .08,
    "candidates": [] if nonplant else [
      {"speciesId": f"species-{i}", "scientificName": f"Test plant{i}",
       "score": .4 - i * .02, "confidence": None} for i in range(5)
    ],
    "confidence": None, "modelVersion": "trained", "productionApproved": False,
  }


@pytest.fixture
def provider(monkeypatch):
  solution = get_solution("plantnet-china")
  model = {
    "architecture": recognition_plantnet.INCREMENTAL_ARCHITECTURE,
    "classCount": 5, "classes": [{"speciesId": f"species-{i}"} for i in range(5)],
    "retainedClassCount": 3, "trainedAdditionalClassCount": 2,
    "speciesSha256": "labels", "presenceGate": {"headSha256": "binary-head"},
  }
  pipeline = SimpleNamespace(
    model=model, version=solution.model_sha256,
    predict=lambda _: pipeline_result(),
  )
  monkeypatch.setattr(recognition_plantnet, "sha256", lambda _: solution.model_sha256)
  monkeypatch.setattr(recognition_plantnet, "PlantNetCascade", lambda *a, **k: pipeline)
  catalog = SimpleNamespace(available=True, get=lambda identifier: {"id": identifier})
  return recognition_plantnet.PlantNetRecognitionProvider(solution, catalog)


def test_online_adapter_includes_binary_gate_and_trained_ids_with_null_confidence(provider):
  result = provider.recognize("synthetic-image")
  assert result["status"] == "requires_review" and result["source"] == "Pl@ntNet"
  assert result["presence_gate"]["passed"] is True
  assert result["presence_gate"]["model_version"] == "binary-head"
  assert result["solution"]["plantPresenceGate"] is True
  assert result["solution"]["supportedPlantCount"] == 5
  assert len(result["candidates"]) == 5
  assert all(row["confidence"] is None for row in result["candidates"])
  assert result["candidates"][0]["species_id"] == "species-0"
  assert len(provider.recognize("synthetic-image", top_k=2)["candidates"]) == 2


def test_binary_rejection_becomes_uncertain_not_failure_or_fake_plant(provider):
  provider.pipeline.predict = lambda _: pipeline_result(True)
  result = provider.recognize("synthetic-image")
  assert result["status"] == "uncertain" and result["reason"] == "nonplant_gate"
  assert result["candidates"] == []
  assert result["presence_gate"]["passed"] is False


def test_dominant_candidate_filtered_but_original_top5_retained(provider):
  raw = pipeline_result()
  for row, score in zip(raw["candidates"], [.981, .0178, .00047, .00026, .00019], strict=True):
    row["score"] = score
  before = deepcopy(raw)
  provider.pipeline.predict = lambda _: raw
  result = provider.recognize("synthetic-image")
  assert len(result["candidates"]) == 1
  assert result["original_prediction"] == before
  assert raw == before
  assert result["candidate_selection"]["returnedCount"] == 1
  assert result["candidate_selection"]["originalCount"] == 5
  assert result["status"] == "requires_review"
  assert result["candidates"][0]["confidence"] is None


def test_hidden_leader_is_not_promoted_to_a_different_successful_species(provider):
  provider.catalog.get = lambda identifier: None if identifier == "species-0" else {"id": identifier}
  result = provider.recognize("synthetic-image")
  assert result["status"] == "uncertain" and result["reason"] == "hidden_leading_candidate"
  assert result["candidates"][0]["species_id"] == "species-0"


@pytest.mark.parametrize("change", ["unknown-id", "nonfinite", "bad-status", "blocked-candidates"])
def test_invalid_pipeline_output_fails_explicitly(provider, change):
  raw = pipeline_result()
  if change == "unknown-id":
    raw["candidates"][0]["speciesId"] = "untrained"
  elif change == "nonfinite":
    raw["candidates"][0]["score"] = float("nan")
  elif change == "bad-status":
    raw["status"] = "random"
  else:
    raw["status"] = "nonplant"
  provider.pipeline.predict = lambda _: deepcopy(raw)
  with pytest.raises(ValueError):
    provider.recognize("synthetic-image")


def test_factory_never_loads_bioclip_for_plantnet(monkeypatch):
  sentinel = object()

  def forbidden(*args, **kwargs):
    pytest.fail("BioCLIP must not initialize for the Pl@ntNet solution.")

  monkeypatch.setattr(catalog_recognition, "CatalogRecognitionProvider", forbidden)
  monkeypatch.setattr(recognition_plantnet, "PlantNetRecognitionProvider", lambda *a, **k: sentinel)
  assert create_provider("plantnet-china", "catalog") is sentinel


def test_active_592_model_and_explicit_431_rollback_are_pinned_separately():
  active = get_solution("plantnet-china")
  rollback = get_solution("plantnet-china-431")
  assert active.version == "20260924.1"
  assert active.model_directory == "plantnet-china-replay-v2"
  assert active.model_sha256 == (
    "25279397d94533a437ef0110eb4bbc0817d510ebff1ce394e211fdb64b6e210c"
  )
  assert rollback.version == "20260923.1"
  assert rollback.model_directory == "plantnet-china-incremental-v1"
  assert rollback.model_sha256 == (
    "4d750ba0a6b83b8498b7739e2680c367f5af6284edc99aa3a6c9dd7257d8967e"
  )
  assert active.experimental and rollback.experimental


def test_registered_model_pin_checked_before_loading_weights(monkeypatch):
  monkeypatch.setattr(recognition_plantnet, "sha256", lambda _: "changed")
  with pytest.raises(ValueError, match="checksum"):
    recognition_plantnet.PlantNetRecognitionProvider(
      get_solution("plantnet-china"), SimpleNamespace(available=True),
    )


def test_health_reports_both_stages_and_actual_loaded_class_count(provider):
  runtime = RecognitionRuntime(SimpleNamespace(), start=False, solution_id="plantnet-china")
  runtime.provider = provider
  info = runtime.solution_info()
  assert info["id"] == "plantnet-china"
  assert info["plantPresenceGate"] is True
  assert info["trainedAdditionalPlantCount"] == 2
  assert info["candidateSelection"]["policy"] == "top-score-ratio-v1"
