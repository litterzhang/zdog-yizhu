"""Synthetic cascade contracts; no downloads, model inference or private records."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from yizhu_server import catalog_recognition, recognition_plantclef, recognition_rerank
from yizhu_server.recognition_cascade import CascadeProvider, crop_boxes, fuse_views
from yizhu_server.recognition_rerank import reranked_result
from yizhu_server.recognition_solutions import create_provider, get_solution

Image = pytest.importorskip("PIL.Image")
SOLUTION = "bioclip-multiview-plantclef"


def recall(ids, reason=None):
  return {
    "source": "BioCLIP", "revision": "test", "label_version": "test-catalog",
    "status": "uncertain" if reason else "requires_review", "reason": reason,
    "candidates": [
      {"species_id": name, "scientific_name": f"Example {name}", "kind": "plant",
       "score": 0.9 - i * 0.001, "confidence": None}
      for i, name in enumerate(ids)
    ],
  }


def secondary(ids):
  return {"source": "PlantCLEF", "revision": "test",
          "candidates": [
            {"species_id": identifier, "kind": "plant" if identifier else "unmapped",
             "scientific_name": f"Example {identifier}", "score": 0.9 - i * 0.1,
             "confidence": None}
            for i, identifier in enumerate(ids)
          ]}


def make_provider(*, scores=None, fallback_ids=("a", "b"), visible=("a", "b", "c"), base_reason=None):
  calls = []
  originals = [recall(["a", "b"], base_reason), recall(["a", "b"]), recall(["a", "b"])]
  remaining = iter(originals)
  scores = scores or {"a": 0.8, "b": 0.6}

  def retrieve(image, *, limit):
    calls.append(("retrieve", image.size, limit))
    return next(remaining)

  def rerank(image, result, *, view):
    calls.append(("rerank", view))
    return reranked_result(result, [scores[c["species_id"]] for c in result["candidates"]])

  def fallback(image, *, top_k):
    calls.append(("plantclef", top_k))
    return secondary(fallback_ids)

  provider = CascadeProvider(
    get_solution(SOLUTION), SimpleNamespace(retrieve=retrieve), SimpleNamespace(rerank=rerank),
    SimpleNamespace(recognize=fallback),
    SimpleNamespace(get=lambda identifier: {"id": identifier} if identifier in visible else None),
  )
  return provider, calls, originals


@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (300, 300)])
def test_crops_are_bounded_square_deterministic_and_overlapping(size):
  boxes = crop_boxes(*size)
  assert len(boxes) == 2
  assert crop_boxes(*size) == boxes
  for left, top, right, bottom in boxes:
    assert 0 <= left < right <= size[0]
    assert 0 <= top < bottom <= size[1]
    assert right - left == bottom - top == int(min(size) * 0.8)
  assert crop_boxes(20, 20) == []


def test_rrf_protects_all_whole_candidates_and_bounds_new_pool():
  ids = [f"p{i}" for i in range(50)]
  views = [
    {"id": "whole", "result": recall(ids)},
    {"id": "left", "result": recall([f"new{i}" for i in range(50)])},
    {"id": "right", "result": recall(["new49"] + [f"other{i}" for i in range(49)])},
  ]
  before = copy.deepcopy(views)
  combined = fuse_views(views)
  assert [r["species_id"] for r in combined["candidates"][:50]] == ids
  assert len(combined["candidates"]) == 100
  assert combined["candidates"][50]["species_id"] == "new49"
  assert len({r["species_id"] for r in combined["candidates"]}) == 100
  assert views == before


def test_rejected_crop_cannot_add_candidates():
  result = fuse_views([
    {"id": "whole", "result": recall(["a"])},
    {"id": "crop", "result": recall(["untrusted"], "nonplant_leading_candidate")},
  ])
  assert [c["species_id"] for c in result["candidates"]] == ["a"]


def test_primary_agreement_skips_fallback_and_preserves_raw_results():
  provider, calls, originals = make_provider()
  before = copy.deepcopy(originals)
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  assert result["status"] == "requires_review"
  assert result["candidates"][0]["species_id"] == "a"
  assert not any(c[0] == "plantclef" for c in calls)
  assert len([c for c in calls if c[0] == "retrieve"]) == 3
  assert result["solution"]["id"] == SOLUTION
  assert result["cascade"]["calibrated"] is False
  assert result["cascade"]["fallback_gate"]["called"] is False
  assert all(c["confidence"] is None for c in result["candidates"])
  assert originals == before


@pytest.mark.parametrize("reason", [
  "nonplant_leading_candidate", "insufficient_visual_detail", "unresolved_reference_taxon",
])
def test_whole_image_abstention_stops_all_later_models(reason):
  provider, calls, _ = make_provider(base_reason=reason)
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  assert result["status"] == "uncertain"
  assert result["reason"] == reason
  assert len(calls) == 1
  assert result["cascade"]["plantclef"] is None


def test_primary_disagreement_calls_secondary_and_records_both_stages():
  provider, calls, _ = make_provider(scores={"a": 0.5, "b": 0.8})
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  assert calls[-1] == ("plantclef", 20)
  assert result["status"] == "requires_review"
  assert result["candidates"][0]["species_id"] == "a"
  assert result["cascade"]["primary"]["candidates"][0]["species_id"] == "b"
  assert result["cascade"]["primary_gate"]["accepted"] is False
  assert result["cascade"]["fallback_gate"]["accepted"] is True


def test_new_fallback_candidate_is_suggestion_not_uncalibrated_success():
  provider, _, _ = make_provider(scores={"a": 0.5, "b": 0.8}, fallback_ids=("c", "a"))
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  assert result["status"] == "uncertain"
  assert result["reason"] == "cascade_no_agreement"
  assert [c["species_id"] for c in result["candidates"]] == ["b", "c", "a"]
  assert result["cascade"]["plantclef"]["candidates"][0]["score"] == 0.9
  assert all(c["confidence"] is None for c in result["candidates"])


@pytest.mark.parametrize("fallback_ids,visible", [
  ((None, "a"), ("a", "b")),
  (("a", "b"), ("a",)),  # Hidden primary leader may not be silently replaced.
])
def test_unmapped_or_hidden_leading_species_cannot_be_promoted(fallback_ids, visible):
  provider, _, _ = make_provider(
    scores={"a": 0.5, "b": 0.8}, fallback_ids=fallback_ids, visible=visible,
  )
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  assert result["status"] == "uncertain"
  assert result["cascade"]["fallback_gate"]["accepted"] is False


def test_inference_error_is_not_misrepresented_as_taxonomic_uncertainty():
  provider, calls, _ = make_provider(scores={"a": 0.5, "b": 0.8})

  def broken(*args, **kwargs):
    raise RuntimeError("synthetic model error")

  provider.fallback.recognize = broken
  with Image.new("RGB", (300, 200)) as image, pytest.raises(RuntimeError):
    provider.recognize(image)
  assert not any(call[0] == "external" for call in calls)


def test_invalid_primary_and_secondary_scores_fail_closed():
  provider, _, originals = make_provider()
  originals[0]["candidates"][0]["score"] = float("nan")
  with Image.new("RGB", (300, 200)) as image, pytest.raises(ValueError):
    provider.recognize(image)
  provider, _, _ = make_provider(scores={"a": 0.5, "b": 0.8})
  bad = secondary(["c"])
  bad["candidates"][0]["score"] = float("inf")
  provider.fallback.recognize = lambda *a, **k: bad
  with Image.new("RGB", (300, 200)) as image, pytest.raises(ValueError):
    provider.recognize(image)


def test_new_factory_checks_fallback_assets_before_loading_primary_models(monkeypatch):
  calls = []

  def fallback(catalog, **kwargs):
    calls.append(("plantclef", catalog, kwargs))
    return object()

  def retrieval(catalog, **kwargs):
    calls.append(("retrieval", catalog, kwargs))
    return object()

  def reranker(**kwargs):
    calls.append(("reranker", kwargs))
    return object()

  monkeypatch.setattr(recognition_plantclef, "PlantCLEFProvider", fallback)
  monkeypatch.setattr(catalog_recognition, "CatalogRecognitionProvider", retrieval)
  monkeypatch.setattr(recognition_rerank, "BioClipReranker", reranker)
  provider = create_provider(SOLUTION, "catalog")
  assert isinstance(provider, CascadeProvider)
  assert [c[0] for c in calls] == ["plantclef", "retrieval", "reranker"]
  assert calls[-1][1]["text_cache_path"].name == "text-vectors.sqlite3"

  calls.clear()

  def missing(*args, **kwargs):
    raise ValueError("missing verified PlantCLEF")

  monkeypatch.setattr(recognition_plantclef, "PlantCLEFProvider", missing)
  with pytest.raises(ValueError, match="PlantCLEF"):
    create_provider(SOLUTION, "catalog")
  assert calls == []


@pytest.mark.parametrize("value", [0, 21, 100, True, 1.5])
def test_cascade_rejects_unbounded_client_candidate_count(value):
  provider, calls, _ = make_provider()
  with pytest.raises(ValueError):
    provider.recognize(None, top_k=value)
  assert calls == []


def test_cascade_result_persists_uncertainty_stages_and_manual_correction(tmp_path, national_names):
  from yizhu_server.catalog import PlantCatalog
  from yizhu_server.catalog_import import build_catalog
  from yizhu_server.journal_store import JournalStore

  path = tmp_path / "catalog"
  build_catalog([
    [["Plantae", "Test", "Test", "Test", "Test", genus, epithet], ""]
    for genus, epithet in (("Ginkgo", "biloba"), ("Jasminum", "sambac"), ("Oxalis", "corniculata"))
  ], path, chinese_names=national_names)
  provider, _, _ = make_provider(scores={"a": 0.5, "b": 0.8}, fallback_ids=("c", "a"))
  with Image.new("RGB", (300, 200)) as image:
    result = provider.recognize(image)
  ids = {"a": "bioclip:plant-001", "b": "bioclip:plant-002", "c": "bioclip:plant-024"}
  # Replace fixture identifiers throughout the synthetic provider response.
  def mapped(value):
    if isinstance(value, list):
      return [mapped(v) for v in value]
    if isinstance(value, dict):
      return {key: ids.get(v, v) if key == "species_id" else mapped(v) for key, v in value.items()}
    return value

  result = mapped(result)
  before = copy.deepcopy(result)
  store = JournalStore(tmp_path / "private", catalog=PlantCatalog(path))
  with store.transaction() as db:
    store.put(db, "alice", "recognitions", {
      "id": "record", "ownerId": "alice", "createdAt": "test", "status": "running",
      "confirmedSpeciesId": None, "rawResult": None,
    })
  store.finish("alice", "record", result=result)
  saved = store.snapshot("alice")["recognitions"][0]
  assert saved["status"] == "uncertain"
  assert len(saved["rawResult"]["candidates"]) == 3
  assert all(r["confidence"] is None for r in saved["rawResult"]["candidates"])
  assert saved["rawResult"]["original"]["cascade"]["plantclef"] is not None
  assert saved["rawResult"]["original"]["solution"]["id"] == SOLUTION
  store.confirm("alice", "record", ids["c"])
  assert store.snapshot("alice")["recognitions"][0]["rawResult"] == saved["rawResult"]
  assert store.snapshot("bob")["recognitions"] == []
  assert result == before


def test_offline_cli_selects_new_solution_without_changing_server_settings(monkeypatch, capsys):
  from yizhu_server import recognition_cli

  calls = []
  catalog = SimpleNamespace(metadata={"labelCount": 100})
  monkeypatch.setattr(recognition_cli, "PlantCatalog", lambda: catalog)

  def create(identifier, reference, *, threads):
    calls.append((identifier, reference, threads))
    return SimpleNamespace(recognize=lambda image: recall(["a", "b"]))

  monkeypatch.setattr(recognition_cli, "create_provider", create)
  monkeypatch.setattr(recognition_cli, "case_image", lambda *a: SimpleNamespace(close=lambda: None))
  monkeypatch.setattr(recognition_cli, "version", lambda _: "synthetic")
  monkeypatch.setattr(sys, "argv", [
    "yizhu-recognize", "synthetic.jpg", "--solution", SOLUTION, "--repeat", "1", "--threads", "1",
  ])
  recognition_cli.main()
  report = json.loads(capsys.readouterr().out)
  assert report["solution"] == SOLUTION
  assert report["weights_sha256"] is None
  assert calls == [(SOLUTION, catalog, 1)]


def test_offline_solution_cannot_mix_custom_labels(monkeypatch):
  from yizhu_server import recognition_cli

  monkeypatch.setattr(sys, "argv", [
    "yizhu-recognize", "synthetic.jpg", "--solution", SOLUTION, "--labels", "custom.json",
  ])
  with pytest.raises(SystemExit) as error:
    recognition_cli.main()
  assert error.value.code == 2
