import importlib.util
from collections import defaultdict
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build-plant-candidates.py"
spec = importlib.util.spec_from_file_location("plant_candidates", SCRIPT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def item(name="Ginkgo biloba", identifier="ginkgo", visible=True):
  return {"speciesId": identifier, "scientificName": name,
          "chineseName": "银杏", "isVisible": visible}


def record(identifier="F20:1", **changes):
  return {
    "recordId": identifier, "scientificNameRaw": "Ginkgo biloba L.",
    "chineseNameRaw": "银杏", "sourceRank": None, "sourceWork": "beijing-native",
    "sourceRegion": "华北", "frequency": "未明确", "sourceType": "native-recommendation",
    "suitableUses": ["城市"], **changes,
  }


@pytest.mark.parametrize("note,expected", [
  ("好种", "未明确"), ("好种，暂不按照POWO处理", "未明确"),
  ("常见，生于路旁", "常见"), ("极常见，生于草地", "极常见"),
  ("多见", "多见"), ("极少见，仅见一处", "极少见"),
  ("不常见", "未明确"), ("近年未见，文献记载", "近年未见"),
  (None, "未明确"), ("少见，旧文献记载常见", "少见"),
])
def test_frequency_is_explicit_not_good_taxonomy_or_substring(note, expected):
  assert extractor.frequency(note) == expected


@pytest.mark.parametrize("raw,expected", [
  ("Ginkgo\nbiloba\nL.", "Ginkgo biloba"),
  ("GinkgobilobaL.", "Ginkgo biloba"),
  ("Larix\nprincipis-ru\npprechtii\nMayr.", "Larix principis-rupprechtii"),
])
def test_pdf_whitespace_repair_requires_exact_catalog_prefix(raw, expected):
  compact = defaultdict(list, {extractor.compact(expected): [item(expected)]})
  result = extractor.classify_name(record(scientificNameRaw=raw), compact)
  assert result["match"]["scientificName"] == expected


@pytest.mark.parametrize("rank", ["品种", "变种", "亚种", "变型"])
def test_non_species_source_ranks_are_retained_not_parent_collapsed(rank):
  result = extractor.classify_name(record(sourceRank=rank), {"Ginkgobiloba": [item()]})
  assert result["key"] is None and result["match"] is None
  assert result["rank"] == rank


@pytest.mark.parametrize("name", [
  "Ginkgo biloba var. example", "Ginkgo biloba subsp. example",
  "Ginkgo biloba f. example", "Ginkgo biloba 'Cultivar'",
])
def test_infraspecific_or_cultivar_text_never_becomes_parent_species(name):
  result = extractor.classify_name(record(scientificNameRaw=name), {"Ginkgobiloba": [item()]})
  assert result["key"] is None


def test_hybrid_not_forced_to_nonhybrid_or_counted_approved():
  result = extractor.classify_name(
    record(scientificNameRaw="Populus × xiaohei T. S. Hwang"), {},
  )
  assert result["rank"] == "杂交待核验" and result["match"] is None


def test_longer_epithet_is_not_truncated_to_shorter_known_name():
  result = extractor.classify_name(
    record(scientificNameRaw="Ginkgo bilobata L."), {"Ginkgobiloba": [item()]},
  )
  assert result["match"] is None
  assert result["name"] == "Ginkgo bilobata"


def test_ambiguous_catalog_names_are_not_assigned_arbitrary_species_id():
  result = extractor.classify_name(
    record(), {"Ginkgobiloba": [item(identifier="one"), item(identifier="two")]},
  )
  assert result["match"] is None


def test_dedup_keeps_provenance_and_counts_publication_only_once():
  records = [
    record(),
    record("F24:wild:1", sourceWork="shanghai-survey", sourceRegion="华东", frequency="常见"),
    record("F24:cultivated:1", sourceWork="shanghai-survey", sourceRegion="华东", frequency="少见"),
  ]
  pool, excluded = extractor.make_pool(records, {"Ginkgo biloba": [item()]}, {})
  assert len(pool) == 1 and not excluded
  assert len(pool[0]["sourceRecordIds"]) == 3
  assert len(pool[0]["sourceWorks"]) == 2
  assert pool[0]["frequencies"] == {"未明确": 1, "常见": 1, "少见": 1}
  assert pool[0]["trainingApproved"] is False


def test_chinese_product_name_is_hint_only_not_merged_with_existing_species():
  rows = [
    record(),
    record("F06:proposed:1", scientificNameRaw="", sourceType="horticulture-standard-proposal"),
  ]
  pool, _ = extractor.make_pool(rows, {"Ginkgo biloba": [item()]}, {"银杏": [item()]})
  assert len(pool) == 2
  hinted = next(row for row in pool if row["catalogMatch"] is None)
  assert hinted["nameOnlyCatalogHints"] == [item()]
  assert all(not row["trainingApproved"] for row in pool)


def test_hidden_or_unmatched_species_are_not_dropped():
  pool, _ = extractor.make_pool(
    [record(), record("new", scientificNameRaw="Unknown example")],
    {"Ginkgo biloba": [item(visible=False)]}, {},
  )
  assert len(pool) == 2
  assert any(row["catalogMatch"] and not row["catalogMatch"]["isVisible"] for row in pool)


def test_native_presence_alone_is_not_priority():
  pool, _ = extractor.make_pool(
    [record(suitableUses=["荒山"], frequency="极少见")], {"Ginkgo biloba": [item()]}, {},
  )
  assert pool[0]["priorityReview"] is False


def test_pinned_source_checksum_is_required(tmp_path):
  path = tmp_path / "X.pdf"
  path.write_bytes(b"not-the-published-file")
  with pytest.raises(ValueError, match="checksum"):
    extractor.checked_file(tmp_path, {"id": "X", "directFetchCheck": {"sha256": "0" * 64}})


def test_existing_output_refused_before_sources_are_opened(tmp_path):
  with pytest.raises(ValueError, match="Output exists"):
    extractor.build(tmp_path, tmp_path / "missing", tmp_path / "missing-db", tmp_path)


def test_seed_gap_report_does_not_treat_partial_names_or_hints_as_confirmed_taxa():
  seeds = [
    {"candidateId": "1", "proposedName": "银杏", "proposedScene": "街道",
     "catalogNameMatches": [item()]},
    {"candidateId": "2", "proposedName": "unknown", "proposedScene": "室内",
     "catalogNameMatches": []},
    {"candidateId": "3", "proposedName": "杏", "proposedScene": "果园",
     "catalogNameMatches": []},
  ]
  pool, _ = extractor.make_pool(
    [record(frequency="常见")], {"Ginkgo biloba": [item()]}, {},
  )
  report = extractor.compare_seeds(seeds, pool)
  assert [row["status"] for row in report] == ["taxon-label-found", "gap", "gap"]
  assert report[0]["hasRegionalFrequencyEvidence"] is True
  assert all(not row["trainingApproved"] for row in report)


def test_verified_indoor_list_is_not_regional_abundance_or_automatic_taxon():
  pool, _ = extractor.make_pool([
    record("F26:1", scientificNameRaw="", chineseNameRaw="绿萝", suitableUses=["室内观赏"],
           sourceType="institutional-indoor-common-name-list", frequency="未明确"),
  ], {}, {})
  assert pool[0]["priorityReview"] is True
  assert pool[0]["catalogMatch"] is None
  assert pool[0]["trainingApproved"] is False
  assert pool[0]["frequencies"] == {"未明确": 1}
