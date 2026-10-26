"""Tiny synthetic reference tables; no production database, network, or weights."""

import json

import pytest

from yizhu_server.app import create_app
from yizhu_server.catalog import PlantCatalog, stable_id
from yizhu_server.catalog_import import build_catalog
from yizhu_server.catalog_recognition import ranked_result
from yizhu_server.colchina_names import build_national_index
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.journal_store import JournalStore
from yizhu_server.settings import Settings


def label(genus, epithet, *, order="Testales", family="Testaceae", kingdom="Plantae"):
    return [[kingdom, "Testophyta", "Testopsida", order, family, genus, epithet], ""]


@pytest.fixture
def catalog(tmp_path, national_names):
    names = [
        label("Ginkgo", "biloba", order="Ginkgoales", family="Ginkgoaceae"),
        label("Oxalis", "corniculata", order="Oxalidales", family="Oxalidaceae"),
        # Scientific name repeated under another hierarchy, not another species.
        label("Oxalis", "corniculata", order="Oxalidales", family="Alternateaceae"),
        label("Nonplant", "example", kingdom="Animalia"),
    ] + [label("Testgenus", f"sample{i:02}") for i in range(35)] + [label("Hidden", "plant", order="Hiddenales")]
    path = tmp_path / "catalog.sqlite3"
    translations = build_national_index([
        {"rowNumber": i + 2, "界拉丁名": "Plantae", "物种拉丁名": f"Testgenus sample{i:02}",
         "物种中文名": f"测试草{i}"} for i in range(35)
    ])
    build_catalog(names, path, chinese_names={**national_names, **translations}, growth_forms={
        "Oxalis corniculata": {"form": "herb", "sources": [{"title": "Test source", "url": "https://example.org"}]},
    })
    return PlantCatalog(path)


def test_import_preserves_legacy_ids_and_more_than_24_species(catalog):
    assert catalog.metadata["plantCount"] == 38
    assert catalog.metadata["labelCount"] == 40
    assert catalog.get("bioclip:plant-001")["commonName"] == "银杏"
    assert catalog.get("bioclip:plant-024")["taxonomy"]["family"] == "Oxalidaceae"
    assert catalog.get("bioclip:plant-024")["taxonomyAlternatives"][0]["family"] == "Alternateaceae"
    assert catalog.get("bioclip:plant-024")["sources"][0]["version"] == catalog.metadata["version"]


def test_browse_order_family_genus_species_and_growth_form(catalog):
    orders = catalog.children(growth_form="herb")
    assert orders["total"] == 1
    assert orders["items"][0]["name"] == "Oxalidales"
    families = catalog.children(parent=orders["items"][0]["id"])
    assert [row["name"] for row in families["items"]] == ["Oxalidaceae"]
    genera = catalog.children(parent=families["items"][0]["id"])
    assert genera["items"][0]["rank"] == "genus"
    species = catalog.search(parent=genera["items"][0]["id"])
    assert species["items"][0]["scientificName"] == "Oxalis corniculata"
    assert catalog.search(growth_form="unknown")["total"] == 36
    assert catalog.search(growth_form="tree")["total"] == 0  # Never guess based on name.


def test_search_pagination_and_literal_input(catalog):
    first = catalog.search(q="Testgenus", limit=20)
    second = catalog.search(q="Testgenus", limit=20, offset=20)
    assert first["total"] == 35
    assert len(first["items"]) == 20 and len(second["items"]) == 15
    assert not ({row["id"] for row in first["items"]} & {row["id"] for row in second["items"]})
    assert catalog.search(q="银杏")["items"][0]["id"] == "bioclip:plant-001"
    assert catalog.search(q="ginkgo BILOBA")["total"] == 1
    assert catalog.search(q="%")["total"] == 0
    assert catalog.search(q="' OR 1=1 --")["total"] == 0
    with pytest.raises(ValueError):
        catalog.search(limit=5000)
    with pytest.raises(ValueError):
        catalog.children(parent="missing")
    with pytest.raises(ValueError):
        catalog.search(growth_form="invented")


def test_selected_catalog_does_not_expand_journal_to_entire_flora(catalog, tmp_path):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    assert store.snapshot("alice")["species"] == []
    species = catalog.search(q="sample00")["items"][0]
    with store.transaction() as db:
        store.put(db, "alice", "plants", {
            "id": "stable-plant", "ownerId": "alice", "speciesId": species["id"], "createdAt": "test",
        })
    snapshot = store.snapshot("alice")
    assert [item["id"] for item in snapshot["species"]] == [species["id"]]
    assert snapshot["catalog"]["plantCount"] == 38
    assert store.require_species(species["id"]) == species["id"]


def test_provider_deduplicates_synonyms_and_keeps_unrelated_organisms_as_controls(catalog):
    result = ranked_result(catalog, [1, 2, 0], [0.8, 0.7, 0.6], top_k=2)
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["species_id"] == "bioclip:plant-024"
    assert result["candidates"][0]["confidence"] is None
    assert result["status"] == "requires_review"
    assert result["label_version"] == catalog.metadata["version"]
    nonplant = ranked_result(catalog, [3, 0], [0.8, 0.6], top_k=2)
    assert nonplant["status"] == "uncertain"
    assert nonplant["reason"] == "nonplant_leading_candidate"
    blank = ranked_result(catalog, [0], [0.99], top_k=1, insufficient_detail=True)
    assert blank["status"] == "uncertain"  # High similarity is never a safety guarantee.


def test_new_species_results_persist_and_can_be_confirmed_without_changing_original(catalog, tmp_path):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    with store.transaction() as db:
        store.put(db, "alice", "recognitions", {
            "id": "rec", "ownerId": "alice", "status": "running", "rawResult": None,
            "confirmedSpeciesId": None, "createdAt": "test",
        })
    original = ranked_result(catalog, [4, 3], [0.5, 0.4], top_k=2)
    store.finish("alice", "rec", result=original)
    snapshot = store.snapshot("alice")
    rec = snapshot["recognitions"][0]
    assert rec["status"] == "succeeded"
    new_id = original["candidates"][0]["species_id"]
    assert new_id.startswith("tol10m:")
    assert snapshot["species"][0]["scientificName"] == "Testgenus sample00"
    store.confirm("alice", "rec", new_id)
    assert store.snapshot("alice")["recognitions"][0]["rawResult"]["original"] == original


def test_uncertain_nonplant_result_never_becomes_success_just_because_a_plant_is_present(catalog, tmp_path):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    with store.transaction() as db:
        store.put(db, "alice", "recognitions", {
            "id": "rec", "status": "running", "confirmedSpeciesId": None, "createdAt": "test",
        })
    store.finish("alice", "rec", result=ranked_result(catalog, [3, 0], [0.8, 0.6], top_k=2))
    assert store.snapshot("alice")["recognitions"][0]["status"] == "uncertain"


def test_public_catalog_is_bounded_but_journal_stays_private(catalog, tmp_path):
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    runtime = RecognitionRuntime(store, start=False)
    client = create_app(Settings(h5_dist=tmp_path / "public"), recognition_runtime=runtime).test_client()
    assert client.get("/api/catalog").get_json()["plantCount"] == 38
    assert client.get("/api/catalog/taxa?limit=1").get_json()["limit"] == 1
    assert len(client.get("/api/catalog/species?q=Testgenus&limit=2").get_json()["items"]) == 2
    assert client.get("/api/catalog/species/bioclip:plant-001").get_json()["commonName"] == "银杏"
    assert client.get("/api/catalog/species/unknown").status_code == 404
    assert client.get("/api/catalog/species?chineseOnly=1").get_json()["total"] == 37
    for query in ["limit=0", "limit=51", "offset=-1", "offset=NaN", "chineseOnly=unknown", "growthForm=fake", "q=" + "x" * 81]:
        assert client.get(f"/api/catalog/species?{query}").status_code == 400
    assert client.get("/api/journal").status_code == 401


def test_missing_catalog_returns_clear_unavailable_instead_of_claiming_broad_coverage(tmp_path):
    catalog = PlantCatalog(None)
    store = JournalStore(tmp_path / "journal", catalog=catalog)
    runtime = RecognitionRuntime(store, start=False)
    client = create_app(Settings(h5_dist=tmp_path / "public"), recognition_runtime=runtime).test_client()
    assert client.get("/api/catalog").get_json()["available"] is False
    assert client.get("/api/catalog/species").status_code == 503


def test_catalog_import_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "existing"
    path.write_text("keep")
    with pytest.raises(ValueError, match="exists"):
        build_catalog([label("Test", "plant")], path)
    assert path.read_text() == "keep"
    with pytest.raises(ValueError):
        build_catalog([["invalid"]], tmp_path / "bad")


def test_ids_are_independent_of_source_row_order(tmp_path):
    rows = [label("A", "plant"), label("B", "plant")]
    build_catalog(rows, tmp_path / "one")
    build_catalog(list(reversed(rows)), tmp_path / "two")
    first = PlantCatalog(tmp_path / "one").selected([stable_id("tol10m", "A plant"), stable_id("tol10m", "B plant")])
    second = PlantCatalog(tmp_path / "two").selected([stable_id("tol10m", "A plant"), stable_id("tol10m", "B plant")])
    assert json.dumps(first) == json.dumps(second)


def test_unresolved_plant_labels_are_not_exposed_as_confirmable_species(tmp_path):
    path = tmp_path / "catalog"
    build_catalog([
        label("Testgenus", "sp. ABC"), label("Testgenus", "cf. uncertain"),
        label("Testgenus", "spinosus"),
    ], path)
    catalog = PlantCatalog(path)
    assert catalog.metadata["plantCount"] == 1
    assert catalog.metadata["unresolvedPlantLabelCount"] == 2
    assert catalog.search()["items"] == []
    assert catalog.get(stable_id("tol10m", "Testgenus spinosus"), include_hidden=True)["scientificName"] == "Testgenus spinosus"
    assert catalog.plant_indices() == [2]
    assert catalog.labels([0])[0]["kingdom"] == "Plantae"
    assert catalog.labels([0])[0]["species_id"] is None


def test_browse_tree_does_not_duplicate_orders_due_to_historical_class_names(tmp_path, national_names):
    rows = [label("Ginkgo", "biloba"), label("Oxalis", "corniculata")]
    rows[1][0][2] = "DifferentClass"
    build_catalog(rows, tmp_path / "catalog", chinese_names=national_names)
    catalog = PlantCatalog(tmp_path / "catalog")
    orders = catalog.children()["items"]
    assert len(orders) == 1
    assert orders[0]["count"] == 2


def test_older_catalog_can_still_browse_without_silently_claiming_chinese_support(catalog):
    catalog.metadata.pop("chineseNameCount")
    assert catalog.children()["items"][0]["chineseName"] is None
    assert catalog.search(q="Ginkgo")["total"] == 1
    with pytest.raises(ValueError, match="中文名索引"):
        catalog.children(chinese_only=True)
    with pytest.raises(ValueError, match="中文名索引"):
        catalog.search(chinese_only=True)
