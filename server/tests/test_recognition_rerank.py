"""Offline ranking contracts; synthetic references, no user photos or model downloads."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from yizhu_server import recognition_rerank
from yizhu_server.catalog_recognition import CatalogRecognitionProvider
from yizhu_server.recognition_rerank import reranked_result, validate_candidates


def retrieval(status="requires_review"):
    return {
        "model": "first-stage", "revision": "test-v1", "status": status,
        "reason": "nonplant_leading_candidate" if status == "uncertain" else None,
        "candidates": [
            {"species_id": "species-a", "kind": "plant", "scientific_name": "Example a",
             "score": 0.5, "confidence": None},
            {"species_id": "species-b", "kind": "plant", "scientific_name": "Example b",
             "score": 0.4, "confidence": None},
        ],
        "global_leading_candidate": {"name": "control", "score": 0.6},
    }


def test_rerank_keeps_ids_and_original_scores_without_injecting_a_target():
    original = retrieval()
    before = copy.deepcopy(original)
    result = reranked_result(original, [0.1, 0.3])
    assert original == before
    assert [row["species_id"] for row in result["candidates"]] == ["species-b", "species-a"]
    assert result["candidates"][0]["retrieval_rank"] == 2
    assert result["candidates"][0]["retrieval_score"] == 0.4
    assert result["candidates"][0]["score"] == 0.3
    assert all(row["confidence"] is None for row in result["candidates"])
    assert result["first_stage"] == before
    assert result["model"] == recognition_rerank.MODEL_ID
    assert result["scope"] == "retrieved_shortlist_not_full_bioclip2_reference"
    result["first_stage"]["candidates"][0]["score"] = -1
    assert original == before


def test_high_rerank_score_never_overrides_first_stage_abstention():
    result = reranked_result(retrieval("uncertain"), [0.99, 0.1])
    assert result["status"] == "uncertain"
    assert result["reason"] == "nonplant_leading_candidate"
    assert result["first_stage"]["global_leading_candidate"]["name"] == "control"


@pytest.mark.parametrize("scores", [[0.1], [0.1, float("nan")], [float("inf"), 0.2], [True, 0.1]])
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        reranked_result(retrieval(), scores)


def test_references_must_be_bounded_unique_retrieved_plants():
    row = retrieval()["candidates"][0]
    for candidates in [[], [row] * 101, [row, row], [{**row, "kind": "nonplant"}],
                       [{**row, "scientific_name": ""}], [{"kind": "plant"}]]:
        with pytest.raises(ValueError):
            validate_candidates(candidates)
    with pytest.raises(ValueError):
        reranked_result(retrieval("succeeded"), [0.1, 0.2])


def test_ties_preserve_original_rank_and_negative_cosine_scores_are_valid():
    result = reranked_result(retrieval(), [-0.1, -0.1])
    assert [row["species_id"] for row in result["candidates"]] == ["species-a", "species-b"]


def test_live_provider_keeps_five_candidates_by_default():
    provider = CatalogRecognitionProvider.__new__(CatalogRecognitionProvider)
    provider.retrieve = lambda image, limit: {"limit": limit}
    assert provider.recognize("test") == {"limit": 5}
    with pytest.raises(ValueError):
        provider.recognize("test", top_k=50)


def test_wide_offline_retrieval_still_has_a_hard_limit():
    provider = CatalogRecognitionProvider.__new__(CatalogRecognitionProvider)
    for limit in [0, 101, True, 2.5]:
        with pytest.raises(ValueError):
            provider.retrieve("test", limit=limit)


def test_reranker_checks_pinned_weights_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.safetensors"
    path.write_bytes(b"abc")
    monkeypatch.setattr(recognition_rerank, "MODEL_BYTES", 3)
    monkeypatch.setattr(recognition_rerank, "MODEL_SHA256", hashlib.sha256(b"abc").hexdigest())
    recognition_rerank.verify_weights(path)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum"):
        recognition_rerank.verify_weights(path)
    path.write_bytes(b"incomplete")
    with pytest.raises(ValueError):
        recognition_rerank.verify_weights(path)


@pytest.mark.parametrize("rerank", [False, True])
def test_cli_reranking_is_opt_in_and_retains_first_stage(monkeypatch, capsys, rerank):
    from yizhu_server import recognition_cli

    calls = []

    class Provider:
        catalog = SimpleNamespace(metadata={"labelCount": 100})

        def __init__(self, **kwargs):
            pass

        def recognize(self, image):
            calls.append("live-default")
            return retrieval()

        def retrieve(self, image, *, limit):
            calls.append(("shortlist", limit))
            return retrieval("uncertain")

    class Reranker:
        def __init__(self, **kwargs):
            calls.append("reranker")

        def rerank(self, image, first_stage, *, view="official"):
            return reranked_result(first_stage, [0.1, 0.9], view=view)

    monkeypatch.setattr(recognition_cli, "CatalogRecognitionProvider", Provider)
    monkeypatch.setattr(recognition_cli, "case_image", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(recognition_cli, "version", lambda package: "test")
    monkeypatch.setattr(recognition_rerank, "verify_weights", lambda path: calls.append("verify"))
    monkeypatch.setattr(recognition_rerank, "BioClipReranker", Reranker)
    monkeypatch.setattr(sys, "argv", ["yizhu-recognize", "synthetic.jpg", "--repeat", "1"]
                        + (["--rerank"] if rerank else []))
    recognition_cli.main()
    report = json.loads(capsys.readouterr().out)
    if rerank:
        assert calls == ["verify", ("shortlist", 50), "reranker"]
        assert report["cases"][0]["result"]["status"] == "uncertain"
        assert report["cases"][0]["result"]["first_stage"] == retrieval("uncertain")
        assert report["reranker"]["shortlist_limit"] == 50
    else:
        assert calls == ["live-default"]
        assert "reranker" not in report
        assert report["cases"][0]["result"] == retrieval()


def test_cli_rejects_hand_selected_labels_for_reranking(monkeypatch):
    from yizhu_server import recognition_cli

    monkeypatch.setattr(sys, "argv", [
        "yizhu-recognize", "synthetic.jpg", "--labels", "custom.json", "--rerank",
    ])
    with pytest.raises(SystemExit) as error:
        recognition_cli.main()
    assert error.value.code == 2


def test_view_provenance_is_explicit_and_unknown_views_fail():
    result = reranked_result(retrieval(), [0.2, 0.3], view="whole")
    assert result["image_view"] == "whole"
    assert result["text_template_count"] == 3
    with pytest.raises(ValueError):
        reranked_result(retrieval(), [0.2, 0.3], view="invented")
