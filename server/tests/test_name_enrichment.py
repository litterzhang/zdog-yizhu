"""Offline synthetic fixtures only: no user records, source downloads, or live writes."""

import json
import sqlite3
import zipfile
from contextlib import closing

import pytest

from yizhu_server import name_enrichment as enrichment
from yizhu_server.catalog_import import build_catalog
from yizhu_server.catalog_names import apply_aliases
from yizhu_server.colchina_names import build_national_index
from yizhu_server.name_enrichment import (
    SCIDB_SOURCE,
    build_candidates,
    digest,
    load_entities,
    scidb_records,
    scidb_rows,
    wikidata_records,
)


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    rows = [
        [["Plantae", "Testophyta", "Testopsida", "Testales", "Testaceae",
          "Example", epithet], ""]
        for epithet in ("first", "second", "hidden")
    ]
    index = build_national_index([
        {"界拉丁名": "Plantae", "物种拉丁名": f"Example {name}",
         "物种中文名": chinese, "rowNumber": i + 2}
        for i, (name, chinese) in enumerate((("first", "测试木"), ("second", "第二木")))
    ])
    build_catalog(rows, path, chinese_names=index)
    return path


def source_row(scientific="Example first", chinese="测试木", aliases="测试花"):
    return {
        "rowNumber": 2, "学名-scientific name": scientific,
        "中文名-Chinese name": chinese, "别名-variant name": aliases,
        "科名-Family": "Testaceae", "属名-Genus": "Example", "命名人-authors": "Test.",
    }


def candidates(output):
    with closing(sqlite3.connect(output)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM candidates ORDER BY id")]


def claim(value, **extra):
    return {"mainsnak": {"datavalue": {"value": value}}, "rank": "normal", **extra}


def entities():
    return {
        "Q1": {
            "id": "Q1", "lastrevid": 101, "aliases": {"zh": [{"value": "不应导入"}]},
            "claims": {
                "P225": [claim("Example first")],
                "P105": [claim({"id": "Q7432"})],
                "P1843": [claim(
                    {"text": "测试花", "language": "zh"}, id="Q1$statement",
                    references=[{"snaks": {"P248": [{"datavalue": {"value": {"id": "Q2"}}}],
                                           "P304": [{"datavalue": {"value": "107"}}]}}],
                )],
            },
        },
        "Q2": {
            "id": "Q2", "lastrevid": 102,
            "labels": {"zh": {"value": "合成测试文献"}},
            "claims": {"P361": [claim({"id": enrichment.FRPS_SERIES})]},
        },
    }


def test_separate_output_keeps_catalog_ids_visibility_and_names(catalog, tmp_path):
    original = catalog.read_bytes()
    rows = [
        source_row(),
        {**source_row("Example hidden", "待审木", ""), "rowNumber": 3},
        {**source_row("Example first subsp. small", "亚种", ""), "rowNumber": 4},
        {**source_row("Other first", "不同属", ""), "rowNumber": 5},
    ]
    out = tmp_path / "pending.sqlite3"
    summary = build_candidates(catalog, out, scidb_records(rows), [SCIDB_SOURCE])
    assert catalog.read_bytes() == original
    assert summary["productionChanges"] is False
    assert summary["counts"]["matched_hidden_species"] == 1
    assert summary["counts"]["matched_visible_species"] == 1
    items = candidates(out)
    assert {r["review_status"] for r in items} == {"pending"}
    hidden = next(r for r in items if r["name"] == "待审木")
    assert hidden["current_visible"] == 0
    assert "hidden_requires_policy_approval" in json.loads(hidden["flags"])
    assert hidden["current_chinese_name"] is None
    assert next(r for r in items if r["name"] == "亚种")["species_id"] is None
    assert next(r for r in items if r["name"] == "不同属")["match_status"] == "unmatched"


def test_duplicate_aliases_shared_names_and_conflicting_primary(catalog, tmp_path):
    rows = [
        source_row(aliases="共同名、共同名；测试木"),
        {**source_row("Example second", "另一个名称", "共同名"), "rowNumber": 3},
    ]
    out = tmp_path / "pending"
    summary = build_candidates(catalog, out, scidb_records(rows), [SCIDB_SOURCE])
    assert summary["counts"]["names_shared_across_species"] == 1
    items = candidates(out)
    assert len(items) == 4
    shared = [r for r in items if r["name"] == "共同名"]
    assert len({r["species_id"] for r in shared}) == 2
    assert all("shared_name" in json.loads(r["flags"]) for r in shared)
    changed = next(r for r in items if r["name"] == "另一个名称")
    assert "primary_name_difference" in json.loads(changed["flags"])
    assert changed["current_chinese_name"] == "第二木"


def test_stable_candidate_ids_and_no_overwrites(catalog, tmp_path):
    out, other = tmp_path / "one", tmp_path / "two"
    for path in (out, other):
        build_candidates(catalog, path, scidb_records([source_row()]), [SCIDB_SOURCE])
    assert candidates(out) == candidates(other)
    for path in (out, catalog):
        original = path.read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            build_candidates(catalog, path, [], [])
        assert path.read_bytes() == original
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "not-created")
    with pytest.raises(ValueError, match="already exists"):
        build_candidates(catalog, link, [], [])


def test_alias_conflicting_with_other_existing_primary_is_flagged(catalog, tmp_path):
    output = tmp_path / "pending"
    build_candidates(
        catalog, output, scidb_records([source_row(aliases="第二木")]), [SCIDB_SOURCE],
    )
    alias = next(item for item in candidates(output) if item["name"] == "第二木")
    assert "matches_other_catalog_name" in json.loads(alias["flags"])
    assert alias["current_chinese_name"] == "测试木"


def test_source_rows_without_chinese_names_are_not_invented(catalog, tmp_path):
    output = tmp_path / "pending"
    summary = build_candidates(
        catalog, output, scidb_records([source_row(chinese="", aliases="")]), [SCIDB_SOURCE],
    )
    assert not candidates(output)
    assert summary["counts"]["records_without_names"] == 1


def test_failed_source_does_not_publish_partial_output(catalog, tmp_path):
    def records():
        yield from scidb_records([source_row()])
        raise ValueError("source interrupted")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="interrupted"):
        build_candidates(catalog, output, records(), [SCIDB_SOURCE])
    assert not output.exists()
    assert not list(tmp_path.glob("names-*.sqlite3"))


def test_wikidata_only_uses_statements_and_preserves_citation():
    records = list(wikidata_records(entities()))
    assert len(records) == 1
    item = records[0]
    assert item["names"][0]["name"] == "测试花"
    assert item["flags"] == []
    assert item["rank"] == "species"
    ref = item["evidence"]["references"][0]
    assert ref["revision"] == 102
    assert ref["reference"]["snaks"]["P304"][0]["datavalue"]["value"] == "107"


@pytest.mark.parametrize("change, expected", [
    ({"references": []}, "unverified_reference"),
    ({"qualifiers": {"P1": []}}, "qualified_statement"),
])
def test_unverified_or_qualified_statements_are_flagged(change, expected):
    data = entities()
    data["Q1"]["claims"]["P1843"][0].update(change)
    assert expected in list(wikidata_records(data))[0]["flags"]


def test_unknown_reference_is_not_trusted_even_if_it_has_a_title():
    data = entities()
    data["Q2"]["claims"] = {}
    records = list(wikidata_records(data))
    assert records[0]["evidence"]["references"] == []
    assert "unverified_reference" in records[0]["flags"]


def test_deprecated_and_nonchinese_statements_are_skipped():
    data = entities()
    data["Q1"]["claims"]["P1843"][0]["rank"] = "deprecated"
    data["Q1"]["claims"]["P1843"].append(
        claim({"text": "test name", "language": "en"}, id="Q1$english")
    )
    assert list(wikidata_records(data)) == []


def test_ambiguous_or_nonspecies_wikidata_is_not_matched(catalog, tmp_path):
    for i, (property_name, value) in enumerate((
        ("P225", [claim("Example first"), claim("Example second")]),
        ("P105", [claim({"id": "Q34740"})]),
    )):
        data = entities()
        data["Q1"]["claims"][property_name] = value
        out = tmp_path / f"pending-{i}"
        build_candidates(catalog, out, wikidata_records(data), [{"id": "wikidata"}])
        assert candidates(out)[0]["species_id"] is None


def test_compound_and_non_species_names_require_review(catalog, tmp_path):
    data = entities()
    statement = data["Q1"]["claims"]["P1843"][0]
    statement["mainsnak"]["datavalue"]["value"]["text"] = "测试花根，测试花"
    out = tmp_path / "pending"
    build_candidates(catalog, out, wikidata_records(data), [{"id": "wikidata"}])
    assert "compound_or_qualified_name" in json.loads(candidates(out)[0]["flags"])
    assert candidates(out)[0]["review_status"] == "pending"


def test_source_snapshot_hash_and_identity(tmp_path):
    path = tmp_path / "Q1.json"
    path.write_text(json.dumps({"entities": entities()}))
    items, snapshots = load_entities(tmp_path)
    assert items == entities()
    assert snapshots == [{"file": "Q1.json", "sha256": digest(path)}]
    (tmp_path / "Q2.json").write_text(path.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        load_entities(tmp_path)


def test_workbook_rejects_unverified_bytes(tmp_path):
    path = tmp_path / "unknown.xlsx"
    path.write_bytes(b"not the publisher snapshot")
    with pytest.raises(ValueError, match="checksum"):
        list(scidb_rows(path))


@pytest.mark.parametrize("formula", [False, True])
def test_workbook_parser_rejects_formula_or_incomplete_rows(tmp_path, monkeypatch, formula):
    path = tmp_path / "fixture.xlsx"
    headers = "".join(
        f'<c r="{chr(65 + i)}1" t="inlineStr"><is><t>{name}</t></is></c>'
        for i, name in enumerate(enrichment.HEADERS)
    )
    row = '<c r="F2"><f>1+1</f><v>2</v></c>' if formula else ""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{enrichment.NS[1:-1]}"/>')
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<worksheet xmlns="{enrichment.NS[1:-1]}"><sheetData>'
                   f'<row r="1">{headers}</row><row r="2">{row}</row>'
                   '</sheetData></worksheet>')
    monkeypatch.setattr(enrichment, "SCIDB_SIZE", path.stat().st_size)
    monkeypatch.setattr(enrichment, "SCIDB_SHA256", digest(path))
    with pytest.raises(ValueError, match="Formula" if formula else "row count"):
        list(scidb_rows(path))


def test_apply_aliases_updates_search_only_and_preserves_catalog(catalog, tmp_path):
    from yizhu_server.catalog import PlantCatalog

    original = catalog.read_bytes()
    pending, output = tmp_path / "pending", tmp_path / "updated"
    build_candidates(catalog, pending, scidb_records([source_row()]), [SCIDB_SOURCE])
    report = apply_aliases(catalog, pending, output)
    assert catalog.read_bytes() == original
    assert report["aliasesAdded"] == report["changedSpecies"] == 1
    before, after = PlantCatalog(catalog), PlantCatalog(output)
    result = after.search(q="测试花")["items"][0]
    previous = before.get(result["id"])
    assert result["chineseAliases"] == ["测试花"]
    assert result["nameSources"][-1]["license"] == "CC-BY-4.0"
    assert result["nameSources"][-1]["sourceName"] == "测试花"
    for key in previous:
        if key not in {"chineseAliases", "nameSources"}:
            assert previous[key] == result[key]
    assert before.search(q="测试花")["total"] == 0
    assert after.metadata["visiblePlantCount"] == before.metadata["visiblePlantCount"]
    with closing(sqlite3.connect(catalog)) as left, closing(sqlite3.connect(output)) as right:
        for table in ("labels", "taxa", "membership"):
            assert left.execute(f"SELECT * FROM {table}").fetchall() == right.execute(
                f"SELECT * FROM {table}").fetchall()


def test_apply_preserves_hidden_and_conflicting_names(catalog, tmp_path):
    pending, output = tmp_path / "pending", tmp_path / "updated"
    rows = [
        source_row(aliases="第二木"),
        {**source_row("Example hidden", "隐藏木", "隐藏别名"), "rowNumber": 3},
        {**source_row("Example second", "更改首选名", "不匹配科"),
         "rowNumber": 4, "科名-Family": "Otheraceae"},
    ]
    build_candidates(catalog, pending, scidb_records(rows), [SCIDB_SOURCE])
    report = apply_aliases(catalog, pending, output)
    assert report["changedSpecies"] == report["aliasesAdded"] == 0
    with closing(sqlite3.connect(catalog)) as left, closing(sqlite3.connect(output)) as right:
        assert left.execute("SELECT * FROM species").fetchall() == right.execute(
            "SELECT * FROM species").fetchall()


def test_apply_wikidata_requires_sourced_unqualified_statement(catalog, tmp_path):
    data = entities()
    data["Q1"]["claims"]["P1843"].extend([
        claim({"text": "无来源别名", "language": "zh"}, id="Q1$unsourced"),
        claim({"text": "测试花根", "language": "zh"}, id="Q1$root",
              references=data["Q1"]["claims"]["P1843"][0]["references"]),
    ])
    pending, output = tmp_path / "pending", tmp_path / "updated"
    build_candidates(catalog, pending, wikidata_records(data), [
        {"id": "wikidata", "license": "CC0-1.0"},
    ])
    report = apply_aliases(catalog, pending, output)
    assert report["aliasesAdded"] == 1
    with closing(sqlite3.connect(output)) as db:
        item = json.loads(db.execute(
            "SELECT data FROM species WHERE scientific_name='Example first'"
        ).fetchone()[0])
    assert item["chineseAliases"] == ["测试花"]
    assert item["nameSources"][-1]["references"][0]["revision"] == 102


def test_apply_is_idempotent_and_does_not_overwrite(catalog, tmp_path):
    pending, output, repeat = tmp_path / "pending", tmp_path / "updated", tmp_path / "repeat"
    build_candidates(catalog, pending, scidb_records([source_row()]), [SCIDB_SOURCE])
    apply_aliases(catalog, pending, output)
    report = apply_aliases(output, pending, repeat)
    assert report["aliasesAdded"] == report["changedSpecies"] == report["sourcesAdded"] == 0
    assert report["alreadyApplied"] == 1
    with closing(sqlite3.connect(output)) as left, closing(sqlite3.connect(repeat)) as right:
        assert left.execute("SELECT * FROM species").fetchall() == right.execute(
            "SELECT * FROM species").fetchall()
    for target in (catalog, pending, output):
        before = target.read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            apply_aliases(catalog, pending, target)
        assert target.read_bytes() == before


def test_apply_refuses_stale_catalog_and_leaves_no_output(catalog, tmp_path):
    pending, output = tmp_path / "pending", tmp_path / "updated"
    build_candidates(catalog, pending, scidb_records([source_row()]), [SCIDB_SOURCE])
    with sqlite3.connect(catalog) as db:
        db.execute("UPDATE species SET search_text='changed' WHERE scientific_name='Example first'")
    with pytest.raises(ValueError, match="another catalog"):
        apply_aliases(catalog, pending, output)
    assert not output.exists()
    assert not list(tmp_path.glob("catalog-aliases-*"))


def test_apply_rejects_unverified_source_manifest(catalog, tmp_path):
    pending, output = tmp_path / "pending", tmp_path / "updated"
    build_candidates(
        catalog, pending, scidb_records([source_row()]),
        [{**SCIDB_SOURCE, "snapshotSha256": "not-a-verified-file"}],
    )
    report = apply_aliases(catalog, pending, output)
    assert report["changedSpecies"] == 0


def test_reveal_fills_missing_names_preserving_ids_taxonomy_and_existing_names(
    catalog, tmp_path, monkeypatch,
):
  from yizhu_server import catalog_names
  from yizhu_server.catalog import PlantCatalog

  rows = [
    source_row("Example hidden", "新增木"),
    {**source_row("Example first", "不能覆盖"), "rowNumber": 3},
  ]
  rows[0]["科名-Family"] = "Otheraceae"
  monkeypatch.setattr(catalog_names, "scidb_rows", lambda _: iter(rows))
  before = catalog.read_bytes()
  output = tmp_path / "revealed"
  report = catalog_names.reveal_sciencedb(catalog, tmp_path / "source", output, expected_count=1)
  assert catalog.read_bytes() == before
  assert report["revealedSpecies"] == 1
  assert report["visiblePlantCount"] == report["chineseNameCount"] == 3
  assert report["hiddenPlantCount"] == 0
  assert report["taxonomyDifferenceCount"] == 1
  original, updated = PlantCatalog(catalog), PlantCatalog(output)
  added = updated.search(q="新增木")["items"][0]
  old = original.get(added["id"], include_hidden=True)
  assert original.get(added["id"]) is None
  assert added["id"] == old["id"]
  assert added["scientificName"] == old["scientificName"]
  assert added["taxonomy"] == old["taxonomy"]
  assert added["chineseName"] == added["commonName"] == "新增木"
  assert added["chineseNameOrigin"] == "sciencedb"
  assert added["isVisible"] is True
  assert added["nameSources"][-1]["taxonomyDifferences"] == ["family"]
  assert added["nameSources"][-1]["sourceName"] == "新增木"
  assert added["nameSources"][-1]["license"] == "CC-BY-4.0"
  assert updated.search(q="不能覆盖")["total"] == 0
  for item in original.search()["items"]:
    assert updated.get(item["id"]) == item
  # Taxon browsing and the Chinese-only filter must see the same new total.
  assert updated.search(chinese_only=True)["total"] == 3
  assert updated.children()["items"][0]["count"] == 3


def test_reveal_ignores_missing_chinese_and_nonexact_taxa(catalog, tmp_path, monkeypatch):
  from yizhu_server import catalog_names

  rows = [
    source_row("Example hidden", ""),
    source_row("Example hidde", "拼写不同"),
    source_row("Example hidden subsp. minor", "亚种"),
    source_row("Example", "不能用属名"),
  ]
  monkeypatch.setattr(catalog_names, "scidb_rows", lambda _: iter(rows))
  result = catalog_names.reveal_sciencedb(catalog, tmp_path / "source", tmp_path / "out")
  assert result["revealedSpecies"] == 0
  assert result["visiblePlantCount"] == 2
  assert result["hiddenPlantCount"] == 1


def test_reveal_expected_count_failure_publishes_nothing(catalog, tmp_path, monkeypatch):
  from yizhu_server import catalog_names

  monkeypatch.setattr(catalog_names, "scidb_rows", lambda _: iter([
    source_row("Example hidden", "新增木"),
  ]))
  before = catalog.read_bytes()
  output = tmp_path / "out"
  with pytest.raises(ValueError, match="Expected 2"):
    catalog_names.reveal_sciencedb(catalog, tmp_path / "source", output, expected_count=2)
  assert not output.exists()
  assert not list(tmp_path.glob("catalog-primary-*"))
  assert catalog.read_bytes() == before


def test_reveal_is_idempotent_and_refuses_overwrite(catalog, tmp_path, monkeypatch):
  from yizhu_server import catalog_names

  monkeypatch.setattr(catalog_names, "scidb_rows", lambda _: iter([
    source_row("Example hidden", "新增木"),
  ]))
  output, repeat = tmp_path / "out", tmp_path / "repeat"
  catalog_names.reveal_sciencedb(catalog, tmp_path / "source", output)
  report = catalog_names.reveal_sciencedb(output, tmp_path / "source", repeat, expected_count=0)
  assert report["revealedSpecies"] == 0
  with closing(sqlite3.connect(output)) as left, closing(sqlite3.connect(repeat)) as right:
    for table in ("species", "metadata", "labels", "taxa", "membership"):
      assert left.execute(f"SELECT * FROM {table}").fetchall() == right.execute(
        f"SELECT * FROM {table}").fetchall()
  with pytest.raises(ValueError, match="already exists"):
    catalog_names.reveal_sciencedb(catalog, tmp_path / "source", catalog)


def test_reveal_requires_pinned_workbook(catalog, tmp_path):
  from yizhu_server.catalog_names import reveal_sciencedb

  bad = tmp_path / "unverified.xlsx"
  bad.write_bytes(b"not the published workbook")
  with pytest.raises(ValueError, match="checksum"):
    reveal_sciencedb(catalog, bad, tmp_path / "out")
  assert not (tmp_path / "out").exists()


def test_reveal_refuses_duplicate_source_names(catalog, tmp_path, monkeypatch):
  from yizhu_server import catalog_names

  monkeypatch.setattr(catalog_names, "scidb_rows", lambda _: iter([
    source_row("Example hidden", "名称一"), source_row("Example hidden", "名称二"),
  ]))
  with pytest.raises(ValueError, match="Ambiguous"):
    catalog_names.reveal_sciencedb(catalog, tmp_path / "source", tmp_path / "out")
  assert not (tmp_path / "out").exists()
