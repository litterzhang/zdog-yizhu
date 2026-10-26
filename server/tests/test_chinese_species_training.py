from copy import deepcopy
from types import SimpleNamespace

import pytest

from yizhu_server.chinese_species_training import select_classes


def fixture():
  item = {
    "id": "stable", "scientificName": "Example species", "chineseName": "合成草",
    "chineseAliases": ["合成别名"], "taxonomy": {"family": "Exampleaceae"},
  }
  candidate = {
    "candidateId": "source-row", "scientificLabel": "Example species", "rankStatus": "种",
    "catalogMatch": {"speciesId": "stable", "scientificName": "Example species"},
    "sourceRecordIds": ["source:1"], "sourceRegions": ["华东"],
    "sourceTypes": ["survey-wild"], "sourceWorks": ["synthetic-work"],
    "trainingApproved": False,
  }
  rows = [
    {"speciesId": "stable", "scientificName": "Example species", "split": split}
    for split, count in (("train", 20), ("validation", 5), ("test", 5))
    for _ in range(count)
  ]
  return item, candidate, rows


def test_exact_regional_evidence_and_complete_samples_required():
  item, candidate, rows = fixture()
  before = deepcopy(candidate)
  chosen = select_classes(rows, [candidate], SimpleNamespace(get=lambda _: item))
  assert len(chosen) == 1
  assert chosen[0]["speciesId"] == "stable"
  assert chosen[0]["regionalEvidence"]["sourceRecordIds"] == ["source:1"]
  assert "national-commonness" in chosen[0]["scope"]
  assert candidate == before  # Never approves the entire research candidate pool.
  assert select_classes(rows[:-1], [candidate], SimpleNamespace(get=lambda _: item)) == []


@pytest.mark.parametrize("change", [
  "no-region", "foreign-region", "name-only", "wrong-scientific", "wrong-id",
  "no-source", "no-chinese", "duplicate", "cultivar",
])
def test_chinese_name_is_not_evidence_of_occurrence_in_china(change):
  item, candidate, rows = fixture()
  pool = [candidate]
  if change == "no-region":
    candidate["sourceRegions"] = []
  elif change == "foreign-region":
    candidate["sourceRegions"] = ["Europe"]
  elif change == "name-only":
    candidate["sourceTypes"] = ["name-only"]
  elif change == "wrong-scientific":
    candidate["scientificLabel"] = "Different species"
  elif change == "wrong-id":
    candidate["catalogMatch"]["speciesId"] = "another"
  elif change == "no-source":
    candidate["sourceWorks"] = []
  elif change == "no-chinese":
    item["chineseName"] = None
  elif change == "duplicate":
    pool.append(deepcopy(candidate))
  else:
    candidate["rankStatus"] = "品种"
  assert select_classes(rows, pool, SimpleNamespace(get=lambda _: item)) == []
