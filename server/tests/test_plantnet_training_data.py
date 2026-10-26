import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from yizhu_server import plantnet_training_data as additions
from yizhu_server import training_data as data
from yizhu_server.everyday_training_data import TaxonResolver, source_name_taxon


def plant(identifier, name):
    return {
        "speciesId": identifier, "scientificName": name, "chineseName": "植物",
        "inEverydayCandidates": True,
    }


def plan():
    return {
        "catalogSha256": "catalog",
        "pretrainedSupported": [plant("a", "Existing species")],
        "finetuneAdditions": [plant("b", "New species")],
        "counts": {
            "pretrainedChineseSupported": 1, "finetuneAdditions": 1, "combinedTarget": 2,
        },
    }


def payload():
    return {"total_results": 1, "results": [{
        "id": 3, "name": "Accepted species", "rank": "species", "is_active": True,
        "ancestor_ids": [48460, 47126],
        "names": [{
            "name": "New species", "lexicon": "scientific-names", "is_valid": False,
        }],
    }]}


def test_additions_selection_skips_all_existing_members(tmp_path):
    source = tmp_path / "sets.json"
    data.write_json(source, plan())
    document = additions.prepare(tmp_path / "data", source)
    assert document["requestedClassCount"] == 1
    assert document["combinedSolutionTargetCount"] == 2
    assert [p["speciesId"] for p in document["species"]] == ["b"]
    assert document["retainedPretrainedIds"] == ["a"]
    assert document["reservedSourceNames"]["Existing species"] == "a"
    assert additions.prepare(tmp_path / "data", source) == document
    before = plan()
    before["finetuneAdditions"] = before["pretrainedSupported"]
    with pytest.raises(ValueError, match="overlap"):
        additions.union_targets(before)


def test_retained_pool_is_separate_and_cannot_repurpose_existing_B_dataset(tmp_path):
    source = tmp_path / "sets.json"
    data.write_json(source, plan())
    a = additions.prepare(tmp_path / "A", source, scope="retained")
    b = additions.prepare(tmp_path / "B", source)
    assert a["policy"] == additions.RETAINED_POLICY
    assert [r["speciesId"] for r in a["species"]] == ["a"]
    assert a["requestedClassCount"] == 1
    assert a["species"][0]["classRole"] == "retained-pretrained"
    assert a["reservedSourceNames"] == b["reservedSourceNames"]
    assert additions.prepare(tmp_path / "A", source, scope="retained") == a
    for folder, scope in (("A", "additions"), ("B", "retained")):
        with pytest.raises(ValueError, match="scope differs"):
            additions.prepare(tmp_path / folder, source, scope=scope)
    assert additions.prepare(tmp_path / "B", source) == b


def test_source_scientific_name_evidence_is_not_fuzzy_or_common_name_matching():
    result, status = source_name_taxon(payload(), "New species")
    assert status == "matched"
    assert result["sourceScientificName"] == "Accepted species"
    assert result["sourceNameEvidence"]["matchType"] == "listed-scientific-synonym"
    for mutation in [
        {"name": "Similar species"}, {"lexicon": "english"}, {"is_valid": None},
    ]:
        altered = payload()
        altered["results"][0]["names"][0].update(mutation)
        assert source_name_taxon(altered, "New species")[0] is None
    altered = payload()
    altered["results"][0]["matched_term"] = "Other query"
    assert source_name_taxon(altered, "Other query")[0] is None


def test_ambiguous_inactive_nonplant_or_truncated_synonyms_are_rejected():
    for mutation in [
        {"is_active": False}, {"rank": "genus"}, {"ancestor_ids": []},
    ]:
        altered = payload()
        altered["results"][0].update(mutation)
        assert source_name_taxon(altered, "New species")[0] is None
    altered = payload()
    altered["results"].append({**deepcopy(altered["results"][0]), "id": 4})
    altered["total_results"] = 2
    assert source_name_taxon(altered, "New species")[1] == "ambiguous"
    altered["total_results"] = 101
    assert source_name_taxon(altered, "New species")[1] == "search-truncated"


def test_synonym_of_existing_A_cannot_become_new_B_class(tmp_path):
    response = payload()
    response["results"][0]["names"].append({
        "name": "Existing species", "lexicon": "scientific-names", "is_valid": False,
    })

    class Client:
        def json(self, endpoint, params):
            assert params["all_names"] == "true"
            return response

    resolve = TaxonResolver(
        tmp_path, Client(), allow_source_names=True,
        target_names={"Existing species": "a", "New species": "b"},
    )
    assert resolve(plant("b", "New species")) is None
    report = json.loads((tmp_path / "resolution.json").read_text())
    assert report["unresolved"][0]["status"] == "source-name-is-another-target"


def test_public_sample_preserves_catalog_identity_and_source_scientific_evidence():
    details, _ = source_name_taxon(payload(), "New species")
    species = {**plant("b", "New species"), **details}
    observation = {
        "id": 1, "quality_grade": "research", "user": {"id": 1},
        "taxon": {"id": 3, "name": "Accepted species", "rank": "species"},
        "photos": [{
            "id": 1, "license_code": "cc-by", "attribution": "Public photographer",
            "url": f"https://{data.PHOTO_HOST}/photos/1/square.jpg",
        }],
    }
    rows = data.public_samples([observation], species)
    assert rows[0]["scientificName"] == "New species"
    assert rows[0]["sourceScientificName"] == "Accepted species"
    assert rows[0]["speciesId"] == "b"
    del species["sourceNameEvidence"]
    with pytest.raises(ValueError, match="explicit source evidence"):
        data.public_samples([observation], species)


def test_seed_reuses_only_B_and_does_not_modify_source(tmp_path, monkeypatch):
    monkeypatch.setattr(additions.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    root, source = tmp_path / "target", tmp_path / "source"
    sets = tmp_path / "sets.json"
    data.write_json(sets, plan())
    document = additions.prepare(root, sets)
    data.write_json(source / "species.json", {"species": plan()["pretrainedSupported"]
                                           + plan()["finetuneAdditions"]})
    rows = []
    for i, p in enumerate(plan()["pretrainedSupported"] + plan()["finetuneAdditions"], 1):
        photo = source / f"images/{i}.jpg"
        photo.parent.mkdir(exist_ok=True)
        photo.write_bytes(f"synthetic bytes {i}".encode())
        group = data.observer_group(i)
        rows.append({
            **p, "sampleId": f"sample:{i}", "photoId": i, "observationId": i,
            "observerGroup": group, "split": data.group_split(group), "path": f"images/{i}.jpg",
            "imageSha256": data.sha256(photo),
            "differenceHash": hashlib.sha256(str(i).encode()).hexdigest()[:16],
            "license": "cc-by", "attribution": "Synthetic fixture",
        })
        data.write_json(source / f"taxon-matches/{i}.json", {
            "speciesId": p["speciesId"], "scientificName": p["scientificName"],
            "inatTaxonId": i, "status": "matched",
        })
    data.write_json(source / "samples/all.json", {"samples": rows})
    before = (source / "samples/all.json").read_bytes()
    result = additions.seed_public_cache(root, source, document)
    assert result["copiedImages"] == 1 and result["excludedImages"] == 1
    assert (source / "samples/all.json").read_bytes() == before
    assert [r["speciesId"] for r in data.load_samples(root)] == ["b"]
    assert not (root / "images/1.jpg").exists()
    assert (root / "images/2.jpg").read_bytes() == (source / "images/2.jpg").read_bytes()
    assert additions.seed_public_cache(root, source, document) == result


def test_seeding_cannot_overwrite_an_already_running_dataset(tmp_path):
    source, root = tmp_path / "source", tmp_path / "target"
    data.write_json(source / "species.json", {})
    data.write_json(root / "collection.json", {})
    with pytest.raises(ValueError, match="before collection"):
        additions.seed_public_cache(root, source, {"species": []})


def test_priority_changes_scheduling_only_and_keeps_all_targets():
    document = {"policy": "fixed", "species": [
        plant("a", "A species"), plant("b", "B species"), plant("c", "C species"),
    ]}
    before = deepcopy(document)
    ordered = additions.prioritize(document, ["c"])
    assert [row["speciesId"] for row in ordered["species"]] == ["c", "a", "b"]
    assert document == before
    assert ordered["policy"] == "fixed"
    for ids in (["not-in-B"], ["c", "c"]):
        with pytest.raises(ValueError, match="existing target"):
            additions.prioritize(document, ids)
