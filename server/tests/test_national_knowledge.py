import pytest

from yizhu_server.catalog import PlantCatalog, stable_id
from yizhu_server.catalog_import import build_catalog
from yizhu_server.colchina_names import build_national_index
from yizhu_server.species_knowledge import load_knowledge, validate_knowledge


def national_row(name="Magnolia grandiflora", chinese="荷花木兰", **extra):
    return {
        "rowNumber": 2, "物种拉丁名": name, "物种中文名": chinese,
        "界拉丁名": "Plantae", "界中文名": "植物界",
        "科拉丁名": "Magnoliaceae", "科中文名": "木兰科",
        "属拉丁名": "Magnolia", "属中文名": "北美木兰属",
        "审核专家/数据源": "合成测试来源", **extra,
    }


def test_national_exact_rank_matching_and_ambiguous_labels():
    index = build_national_index([national_row()])
    assert index["species", "Magnolia grandiflora"]["name"] == "荷花木兰"
    assert index["family", "Magnoliaceae"]["name"] == "木兰科"
    assert ("species", "Magnolia other") not in index
    ambiguous = build_national_index([national_row(), national_row(chinese="不同名称")])
    assert ("species", "Magnolia grandiflora") not in ambiguous
    assert not build_national_index([national_row(界拉丁名="Animalia")])


def test_national_index_does_not_infer_synonyms_or_supplemental_aliases():
    national = build_national_index([national_row()])
    item = national["species", "Magnolia grandiflora"]
    assert item["aliases"] == []
    assert item["source"]["matchType"] == "exact"
    assert ("species", "Oldgenus grandiflora") not in national
    assert ("species", "Magnolia grandiflor") not in national
    assert ("species", "Magnolia") not in national
    assert item["source"]["matchedScientificName"] == "Magnolia grandiflora"


def test_national_name_does_not_retain_manual_alias(tmp_path):
    index = build_national_index([national_row()])
    rows = [[["Plantae", "Tracheophyta", "Magnoliopsida", "Magnoliales",
              "Magnoliaceae", "Magnolia", "grandiflora"], ""]]
    build_catalog(rows, tmp_path / "catalog", chinese_names=index)
    item = PlantCatalog(tmp_path / "catalog").get("bioclip:plant-010")
    assert item["commonName"] == "荷花木兰"
    assert item["chineseNameOrigin"] == "colchina"
    assert item["chineseAliases"] == []
    assert len(item["nameSources"]) == 1
    assert item["nameSources"][0]["title"] == "中国生物物种名录 2026 · 植物界"
    assert item["scientificName"] == "Magnolia grandiflora"


def test_osmanthus_uses_only_national_name_and_preserves_old_id(tmp_path):
    index = build_national_index([national_row(name="Osmanthus fragrans", chinese="木犀")])
    rows = [[["Plantae", "Tracheophyta", "Magnoliopsida", "Lamiales",
              "Oleaceae", "Osmanthus", "fragrans"], "sweet osmanthus"]]
    build_catalog(rows, tmp_path / "named", chinese_names=index)
    build_catalog(rows, tmp_path / "unnamed")
    named, unnamed = PlantCatalog(tmp_path / "named"), PlantCatalog(tmp_path / "unnamed")
    identifier = "bioclip:plant-009"
    assert named.get(identifier)["commonName"] == "木犀"
    assert named.get(identifier)["chineseAliases"] == []
    assert named.search(q="木犀")["items"][0]["id"] == identifier
    assert named.search(q="桂花")["total"] == 0
    assert unnamed.get(identifier) is None
    assert unnamed.get(identifier, include_hidden=True)["chineseName"] is None
    assert named.labels([0]) == unnamed.labels([0])
    # Even when the database is absent, ID compatibility must not resurrect a name.
    fallback = PlantCatalog(None)
    assert fallback.get(identifier) is None
    assert fallback.selected([identifier])[0]["commonName"] == "Osmanthus fragrans"
    assert fallback.selected([identifier])[0]["chineseAliases"] == []


def test_knowledge_requires_sources_explicit_ai_review_status_and_safe_links():
    valid = {"Example plant": {
        "summary": "测试简介", "isAiGenerated": True, "reviewStatus": "unreviewed",
        "sources": [{"title": "测试出处", "url": "https://example.org/plant"}],
    }}
    assert validate_knowledge(valid) == valid
    for change in [
        {"sources": []}, {"isAiGenerated": "false"}, {"summary": "x" * 1601},
        {"sources": [{"title": "来源", "url": "javascript:alert(1)"}]},
        {"sources": [{"title": "来源", "url": "https://secret:password@example.org"}]},
    ]:
        with pytest.raises(ValueError):
            validate_knowledge({"Example plant": {**valid["Example plant"], **change}})


def test_knowledge_is_saved_in_public_catalog_without_inventing_other_summaries(tmp_path):
    knowledge = load_knowledge()
    assert len(knowledge) == 6
    rows = [
        [["Plantae", "Tracheophyta", "Ginkgoopsida", "Ginkgoales", "Ginkgoaceae", "Ginkgo", "biloba"], ""],
        [["Plantae", "Tracheophyta", "Unknown", "Unknown", "Unknown", "Example", "unknown"], ""],
    ]
    build_catalog(rows, tmp_path / "catalog", knowledge=knowledge)
    catalog = PlantCatalog(tmp_path / "catalog")
    assert catalog.get("bioclip:plant-001", include_hidden=True)["knowledge"] == knowledge["Ginkgo biloba"]
    assert catalog.search(q="Example unknown")["items"] == []
    assert catalog.get(stable_id("tol10m", "Example unknown"), include_hidden=True)["knowledge"] is None
    assert catalog.metadata["knowledgeCount"] == 1
