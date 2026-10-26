"""Server-only selection contracts; no network, model weights, or private records."""

import copy
from types import SimpleNamespace

import pytest

from yizhu_server import catalog_recognition, journal_api, recognition_rerank
from yizhu_server.app import create_app
from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.journal_store import JournalStore
from yizhu_server.recognition_rerank import reranked_result
from yizhu_server.recognition_solutions import (
    DEFAULT_SOLUTION,
    SOLUTIONS,
    SolutionProvider,
    create_provider,
    get_solution,
)
from yizhu_server.settings import Settings


def result(count=8, *, uncertain=False):
    return {
        "source": "BioCLIP", "revision": "synthetic-v1", "is_demo": False,
        "status": "uncertain" if uncertain else "requires_review",
        "reason": "nonplant_leading_candidate" if uncertain else None,
        "candidates": [
            {"species_id": f"synthetic-{i}", "scientific_name": f"Example species{i}",
             "kind": "plant", "score": 0.5 - i * 0.01, "confidence": None}
            for i in range(count)
        ],
    }


@pytest.mark.parametrize("identifier", [
    key for key, value in SOLUTIONS.items()
    if not value.cascade and value.plant_set_source == "catalog-reference"
])
def test_factory_loads_only_selected_stages_and_keeps_client_candidates_bounded(monkeypatch, identifier):
    calls = []
    original = result()
    preserved = copy.deepcopy(original)

    class Retrieval:
        def __init__(self, catalog, *, threads):
            calls.append(("retrieval", catalog, threads))

        def recognize(self, image, *, top_k):
            calls.append(("recognize", top_k))
            return {**original, "candidates": original["candidates"][:top_k]}

        def retrieve(self, image, *, limit):
            calls.append(("retrieve", limit))
            return original

    class Reranker:
        def __init__(self, *, threads):
            calls.append(("reranker", threads))

        def rerank(self, image, first_stage, *, view):
            calls.append(("rerank", view))
            return reranked_result(first_stage, [i * 0.1 for i in range(8)], view=view)

    monkeypatch.setattr(catalog_recognition, "CatalogRecognitionProvider", Retrieval)
    monkeypatch.setattr(recognition_rerank, "BioClipReranker", Reranker)
    provider = create_provider(identifier, "synthetic-catalog")
    response = provider.recognize("synthetic-image")
    assert len(response["candidates"]) == 5
    assert response["source"] == "BioCLIP"
    assert response["solution"] == get_solution(identifier).public_info()
    assert original == preserved
    if identifier == DEFAULT_SOLUTION:
        assert calls == [("retrieval", "synthetic-catalog", 2), ("recognize", 5)]
        assert response["candidates"] == original["candidates"][:5]
    else:
        view = get_solution(identifier).rerank_view
        assert calls == [
            ("retrieval", "synthetic-catalog", 2), ("reranker", 2),
            ("retrieve", 50), ("rerank", view),
        ]
        assert response["candidates"][0]["species_id"] == "synthetic-7"
        assert len(response["reranking_candidates"]) == 8
        assert response["first_stage"] == original
    with pytest.raises(ValueError):
        provider.recognize("synthetic-image", top_k=100)


def test_missing_second_model_never_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(catalog_recognition, "CatalogRecognitionProvider", lambda *a, **k: object())

    def missing(**kwargs):
        raise ValueError("Missing verified weights")

    monkeypatch.setattr(recognition_rerank, "BioClipReranker", missing)
    with pytest.raises(ValueError, match="verified weights"):
        create_provider("bioclip-v2-rerank", None)
    with pytest.raises(ValueError):
        SolutionProvider(get_solution("bioclip-v2-whole"), object())


def test_wrapper_cannot_promote_retrieval_abstention():
    first = result(uncertain=True)
    retrieval = SimpleNamespace(retrieve=lambda *a, **k: first)
    reranker = SimpleNamespace(rerank=lambda *a, **k: result())
    provider = SolutionProvider(get_solution("bioclip-v2-rerank"), retrieval, reranker)
    response = provider.recognize("synthetic-image")
    assert response["status"] == "uncertain"
    assert response["reason"] == "nonplant_leading_candidate"


def test_settings_selects_registered_solution_and_rejects_invalid_values(monkeypatch, tmp_path):
    monkeypatch.delenv("YIZHU_RECOGNITION_SOLUTION", raising=False)
    assert Settings.from_env().recognition_solution == DEFAULT_SOLUTION
    for identifier in SOLUTIONS:
        monkeypatch.setenv("YIZHU_RECOGNITION_SOLUTION", f" {identifier} ")
        assert Settings.from_env().recognition_solution == identifier
    for identifier in ["", "automatic", "arbitrary.module:Provider"]:
        monkeypatch.setenv("YIZHU_RECOGNITION_SOLUTION", identifier)
        with pytest.raises(ValueError, match="YIZHU_RECOGNITION_SOLUTION"):
            Settings.from_env()
    with pytest.raises(ValueError):
        Settings(h5_dist=tmp_path, recognition_solution="unknown")


def test_app_passes_server_selection_to_worker_and_reports_it(monkeypatch, tmp_path):
    original_runtime = RecognitionRuntime

    def runtime_without_thread(store, *, solution_id):
        return original_runtime(store, start=False, solution_id=solution_id)

    monkeypatch.setattr(journal_api, "RecognitionRuntime", runtime_without_thread)
    app = create_app(Settings(
        h5_dist=tmp_path / "public", data_dir=tmp_path / "private",
        recognition_solution="bioclip-v2-whole",
    ))
    response = app.test_client().get("/api/health").get_json()
    assert response["recognition"] == "warming"
    assert response["recognition_solution"]["id"] == "bioclip-v2-whole"
    assert response["recognition_solution"]["experimental"] is True


@pytest.mark.parametrize("fails", [False, True])
def test_runtime_initialization_obeys_selection_without_fallback(monkeypatch, fails):
    calls = []
    store = SimpleNamespace(catalog="catalog", next_job=lambda: None)
    runtime = RecognitionRuntime(store, start=False, solution_id="bioclip-v2-rerank")
    runtime.stop.set()

    def selected(identifier, catalog):
        calls.append((identifier, catalog))
        if fails:
            raise ValueError("Missing model")
        return object()

    monkeypatch.setattr(journal_api, "create_provider", selected)
    runtime.run()
    assert calls == [("bioclip-v2-rerank", "catalog")]
    assert runtime.state == ("failed" if fails else "ready")
    assert runtime.solution_info()["id"] == "bioclip-v2-rerank"
    if fails:
        assert runtime.provider is None


def test_solution_provenance_is_saved_without_changing_journal_contract(tmp_path, national_names):
    catalog_path = tmp_path / "catalog"
    build_catalog([[["Plantae", "Test", "Test", "Test", "Test", "Ginkgo", "biloba"], ""]],
                  catalog_path, chinese_names=national_names)
    store = JournalStore(tmp_path / "private", catalog=PlantCatalog(catalog_path))
    raw = result(1)
    raw["candidates"][0]["species_id"] = "bioclip:plant-001"
    raw["candidates"][0]["scientific_name"] = "Ginkgo biloba"
    provider = SolutionProvider(
        get_solution(DEFAULT_SOLUTION),
        SimpleNamespace(recognize=lambda *a, **k: raw),
    )
    with store.transaction() as db:
        store.put(db, "alice", "recognitions", {
            "id": "record", "ownerId": "alice", "status": "running", "createdAt": "test",
            "confirmedSpeciesId": None, "rawResult": None,
        })
    store.finish("alice", "record", result=provider.recognize("synthetic-image"))
    snapshot = store.snapshot("alice")
    record = snapshot["recognitions"][0]
    assert record["status"] == "succeeded"
    assert record["rawResult"]["source"] == "BioCLIP"
    assert record["rawResult"]["version"] == "synthetic-v1"
    assert record["rawResult"]["candidates"] == [{
        "speciesId": "bioclip:plant-001", "confidence": None, "score": 0.5,
    }]
    saved = copy.deepcopy(record["rawResult"])
    assert saved["original"]["solution"]["id"] == DEFAULT_SOLUTION
    store.confirm("alice", "record", "bioclip:plant-001")
    assert store.snapshot("alice")["recognitions"][0]["rawResult"] == saved
    assert store.snapshot("bob")["recognitions"] == []


def test_client_cannot_select_solution_in_request(tmp_path):
    class Auth:
        configured = True

        def userinfo(self, token):
            return {"sub": "synthetic-user"}

    store = JournalStore(tmp_path / "private", catalog=PlantCatalog(None))
    runtime = RecognitionRuntime(store, start=False)
    client = create_app(Settings(h5_dist=tmp_path / "public"),
                        auth_client=Auth(), recognition_runtime=runtime).test_client()
    response = client.post("/api/recognitions", headers={"Authorization": "Bearer synthetic"},
                           json={"photoId": "photo", "requestId": "request",
                                 "solution": "bioclip-v2-rerank"})
    assert response.status_code == 400
    assert runtime.solution.id == DEFAULT_SOLUTION
