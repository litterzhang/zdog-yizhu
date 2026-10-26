import io
import json
import random
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from yizhu_server import everyday_training_data as everyday
from yizhu_server import training_classifier as classifier
from yizhu_server import training_data as data


def candidate():
    return {
        "candidateId": "candidate:1", "priorityReview": True, "rankStatus": "种",
        "scientificLabel": "Ginkgo biloba", "trainingApproved": False,
        "catalogMatch": {
            "speciesId": "species:1", "scientificName": "Ginkgo biloba", "chineseName": None,
        },
    }


def taxon():
    return {
        "id": 1, "name": "Ginkgo biloba", "rank": "species",
        "is_active": True, "ancestor_ids": [48460, 47126],
    }


def test_fixed_selection_preserves_missing_names_and_does_not_approve_pool():
    row = candidate()
    ignored = {**row, "priorityReview": False}
    before = deepcopy(row)
    selected = everyday.select_targets([row, ignored])
    assert len(selected) == 1 and selected[0]["chineseName"] is None
    assert row == before and row["trainingApproved"] is False


def test_selection_rejects_duplicate_ids_and_name_mismatch():
    row = candidate()
    with pytest.raises(ValueError, match="Duplicate"):
        everyday.select_targets([row, row])
    row["scientificLabel"] = "Other species"
    with pytest.raises(ValueError, match="differs"):
        everyday.select_targets([row])


@pytest.mark.parametrize("change", [
    {"name": "Ginkgo alias"}, {"rank": "genus"}, {"is_active": False},
    {"ancestor_ids": [48460]}, {"id": True},
])
def test_taxon_match_is_exact_active_species_and_plant(change):
    assert everyday.exact_taxon(
        {"results": [{**taxon(), **change}]}, "Ginkgo biloba",
    ) == (None, "no-exact-active-species-match")


def test_taxon_ambiguity_and_truncation_fail_closed():
    assert everyday.exact_taxon({"results": [taxon()]}, "Ginkgo biloba") == (1, "matched")
    assert everyday.exact_taxon(
        {"results": [taxon(), {**taxon(), "id": 2}]}, "Ginkgo biloba",
    ) == (None, "ambiguous")
    assert everyday.exact_taxon(
        {"results": [taxon()], "total_results": 101}, "Ginkgo biloba",
    ) == (None, "search-truncated")


def test_resolver_resumes_from_cache_and_preserves_pause(tmp_path):
    class Client:
        calls = 0

        def json(self, endpoint, params):
            self.calls += 1
            assert endpoint == "taxa" and params["q"] == "Ginkgo biloba"
            return {"results": [taxon()], "total_results": 1}

    client = Client()
    species = everyday.select_targets([candidate()])[0]
    assert everyday.TaxonResolver(tmp_path, client)(species)["inatTaxonId"] == 1
    assert everyday.TaxonResolver(tmp_path, client)(species)["inatTaxonId"] == 1
    assert client.calls == 1
    data.write_json(tmp_path / "paused.json", {"reason": "user pause"})
    with pytest.raises(ValueError, match="paused"):
        everyday.TaxonResolver(tmp_path, client)(species)
    assert (tmp_path / "paused.json").exists()


def test_prepare_refuses_changed_pool_and_wrong_count(tmp_path):
    pool = tmp_path / "pool.json"
    data.write_json(pool, [candidate()])
    with pytest.raises(ValueError, match="Expected the agreed 972"):
        everyday.prepare(tmp_path / "data", pool)
    data.write_json(tmp_path / "data/species.json", {
        "policy": everyday.POLICY, "candidatePoolSha256": "not-this-pool",
    })
    with pytest.raises(ValueError, match="different selection"):
        everyday.prepare(tmp_path / "data", pool)


def test_custom_collection_targets_and_resume_are_not_old_thirty_photo_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(data.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    species = {**everyday.select_targets([candidate()])[0], "inatTaxonId": 1}
    unresolved = {**species, "speciesId": "species:2", "scientificName": "Unresolved species"}
    document = {"species": [species, unresolved]}
    data.write_json(tmp_path / "species.json", document)
    rows = []
    for i, split in enumerate(("train", "train", "validation", "test"), 1):
        rows.append({
            "sampleId": f"sample:{i}", "speciesId": species["speciesId"],
            "scientificName": species["scientificName"], "photoId": i,
            "split": split, "observerGroup": f"observer:{i}", "imageUrl": f"photo:{i}",
        })
    data.write_json(tmp_path / "photo-metadata/1-1.json", {"samples": rows, "hasMore": False})

    class Client:
        calls = 0

        def get(self, url, **kwargs):
            self.calls += 1
            image = Image.frombytes("RGB", (100, 100), random.Random(url).randbytes(30000))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            return stream.getvalue()

    client = Client()
    targets = {"train": 2, "validation": 1, "test": 1}
    def resolve(row):
        return row if row["speciesId"] == "species:1" else None
    report = data.collect(
        tmp_path, client, document, split_targets=targets, resolve_species=resolve,
    )
    assert report["downloadedImages"] == 4 and report["readySpeciesCount"] == 1
    assert report["targetSpecies"] == 2
    assert report["shortfalls"][0]["missing"] == targets
    data.collect(tmp_path, client, document, split_targets=targets, resolve_species=resolve)
    assert client.calls == 4
    assert json.loads((tmp_path / "collection.json").read_text())["splitTargets"] == targets
    with pytest.raises(ValueError, match="policy changed"):
        data.collect(tmp_path, client, document)


def test_collector_preserves_old_paused_dataset(tmp_path):
    data.write_json(tmp_path / "paused.json", {})
    with pytest.raises(ValueError, match="paused"):
        data.collect(tmp_path, None, {"species": []}, split_targets=everyday.TARGETS)


def test_training_respects_new_dataset_minima_without_changing_legacy_default():
    species = everyday.select_targets([candidate()])
    rows = [
        {"speciesId": species[0]["speciesId"], "split": split}
        for split, count in data.SPLIT_TARGETS.items() for _ in range(count)
    ]
    assert classifier.eligible_species(rows, species) == species
    assert classifier.eligible_species(rows, species, everyday.TARGETS) == []
    rows = [
        {"speciesId": species[0]["speciesId"], "split": split}
        for split, count in everyday.TARGETS.items() for _ in range(count)
    ]
    assert classifier.eligible_species(rows, species, everyday.TARGETS) == species
