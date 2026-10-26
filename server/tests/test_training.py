import hashlib
import json
from copy import deepcopy

import pytest

from yizhu_server import training_classifier as classifier
from yizhu_server import training_data as data


def species(index=1):
    return {
        "speciesId": f"species-{index}", "scientificName": f"Plant species{index}",
        "inatTaxonId": index, "chineseName": f"植物{index}",
    }


def observation(index=1, license="cc-by"):
    return {
        "id": index, "quality_grade": "research", "user": {"id": index},
        "taxon": {"id": 1, "rank": "species", "name": "Plant species1"},
        "location": "private-coordinate", "description": "private-description",
        "photos": [{
            "id": index, "license_code": license, "attribution": "Public photographer (CC BY)",
            "url": f"https://{data.PHOTO_HOST}/photos/{index}/square.jpg",
        }],
    }


def test_shortlist_is_ranked_exact_visible_and_not_handwritten():
    counts = [
        {"count": 2, "taxon": {"id": 2, "name": "Plant b", "rank": "species"}},
        {"count": 50, "taxon": {"id": 3, "name": "Unmapped c", "rank": "species"}},
        {"count": 20, "taxon": {"id": 1, "name": "Plant a", "rank": "species"}},
        {"count": 10, "taxon": {"id": 4, "name": "Plant alias", "rank": "species"}},
    ]
    names = {
        "Plant a": {"id": "a", "chineseName": "甲"},
        "Plant alias": {"id": "a", "chineseName": "甲"},
        "Plant b": {"id": "b", "chineseName": "乙"},
    }
    selected, skipped = data.select_species(counts, names, 2)
    assert [row["speciesId"] for row in selected] == ["a", "b"]
    assert skipped == [{"scientificName": "Unmapped c", "count": 50}]


@pytest.mark.parametrize("license", ["cc-by-nc", "cc-by-sa", "cc-by-nd", None, "unknown"])
def test_non_allowlisted_photo_licenses_are_excluded(license):
    assert data.public_samples([observation(license=license)], species()) == []


def test_each_photo_license_checked_not_just_observation_filter():
    sample = observation(license="cc-by-nc")
    permitted = observation()["photos"][0]
    permitted["id"] = 9
    sample["photos"].append(permitted)
    rows = data.public_samples([sample], species())
    assert len(rows) == 1 and rows[0]["photoId"] == 9
    assert rows[0]["imageUrl"].endswith("/photos/9/medium.jpg")
    assert "location" not in rows[0] and "description" not in rows[0]
    assert "user" not in rows[0] and "attribution" in rows[0]


@pytest.mark.parametrize("change", ["hidden", "wrong-species", "rank", "host", "query", "quality"])
def test_unusable_public_samples_rejected(change):
    sample = observation()
    if change == "hidden":
        sample["photos"][0]["hidden"] = True
    elif change == "wrong-species":
        sample["taxon"]["id"] = 999
    elif change == "rank":
        sample["taxon"]["rank"] = "genus"
    elif change == "host":
        sample["photos"][0]["url"] = "https://untrusted.example/photos/1/square.jpg"
    elif change == "query":
        sample["photos"][0]["url"] += "?token=secret"
    else:
        sample["quality_grade"] = "needs_id"
    assert data.public_samples([sample], species()) == []


def test_observer_split_stays_same_across_observations_and_species():
    sample = observation()
    row1 = data.public_samples([sample], species())[0]
    sample["id"] = 12
    sample["taxon"].update({"id": 2, "name": "Plant species2"})
    row2 = data.public_samples([sample], species(2))[0]
    assert row1["observerGroup"] == row2["observerGroup"]
    assert row1["split"] == row2["split"]


def test_global_dedup_rejects_exact_and_up_to_four_changed_bits():
    index = data.DuplicateIndex()
    value = 0xF82A5BC491EE9280
    index.add("a", value)
    assert index.contains("a", value ^ 0xFFFFFFFFFFFFFFFF)
    assert index.contains("b", value ^ ((1 << 2) | (1 << 21) | (1 << 40) | (1 << 61)))
    assert not index.contains("b", value ^ 0xFFFFFFFFFFFFFFFF)


def valid_row():
    row = data.public_samples([observation()], species())[0]
    return {
        **row, "path": "images/1.jpg", "imageSha256": "a" * 64,
        "differenceHash": "f82a5bc491ee9280",
    }


def test_training_rejects_leaks_labels_licenses_and_traversal():
    original = valid_row()
    classifier.validate_samples([original], [species()])
    for field, value in [
        ("path", "../private.jpg"), ("path", "/tmp/private.jpg"),
        ("license", "cc-by-nc"), ("speciesId", "unknown"),
        ("scientificName", "Different plant"),
        ("split", "invalid"), ("attribution", ""),
    ]:
        row = {**original, field: value}
        with pytest.raises(ValueError):
            classifier.validate_samples([row], [species()])
    with pytest.raises(ValueError, match="Duplicate"):
        classifier.validate_samples([original, deepcopy(original)], [species()])


def test_class_eligibility_requires_all_three_splits():
    rows = []
    for split, count in data.SPLIT_TARGETS.items():
        rows.extend({"speciesId": "species-1", "split": split} for _ in range(count))
    assert classifier.eligible_species(rows, [species(), species(2)]) == [species()]
    assert classifier.eligible_species(rows[:-1], [species()]) == []


def test_feature_cache_binds_weights_preprocessing_and_image():
    row = valid_row()
    identity = {"weights": "a", "preprocess": "one"}
    original = classifier.feature_key(row, identity)
    assert classifier.feature_key(row, {**identity, "weights": "b"}) != original
    assert classifier.feature_key(row, {**identity, "preprocess": "two"}) != original
    assert classifier.feature_key({**row, "imageSha256": "c"}, identity) != original


def test_feature_cache_refuses_corruption(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "feature.npy"
    for vector in [[float("nan"), 0], [1, 1], [1, 0, 0]]:
        np.save(path, np.array(vector, dtype=np.float32))
        with pytest.raises(ValueError):
            classifier.read_feature(path, 2)
    np.save(path, np.array([1, 0], dtype=np.float32))
    assert classifier.read_feature(path, 2).tolist() == [1, 0]


def test_linear_head_learns_synthetic_features_without_using_test_for_selection():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[1., 0.], [0., 1.]] * 20)
    labels = torch.tensor([0, 1] * 20)
    head, fit = classifier.fit_head(
        {"train": (x, labels), "validation": (x, labels)}, 2, epochs=60,
    )
    assert fit["epochsRun"] <= 60
    assert classifier.metrics(head(x), labels, 2)["top1"] == 1.0


def test_offline_training_exports_safe_loadable_weights_and_provenance(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    root = tmp_path / "data"
    output = tmp_path / "model"
    root.mkdir()
    classes = [species(), species(2)]
    data.write_json(root / "species.json", {"species": classes})
    # Synthetic fixtures exercise the pipeline, not real recognition accuracy.
    rows, used = [], 0
    for item in classes:
        for split, count in data.SPLIT_TARGETS.items():
            for _ in range(count):
                while True:
                    used += 1
                    group = data.observer_group(used)
                    if data.group_split(group) == split:
                        break
                row = {
                    **valid_row(), "sampleId": str(used), "photoId": used,
                    "observationId": used, "observerGroup": group, "split": split,
                    "speciesId": item["speciesId"], "scientificName": item["scientificName"],
                    "imageSha256": hashlib.sha256(str(used).encode()).hexdigest(),
                    "differenceHash": hashlib.sha256(f"image:{used}".encode()).hexdigest()[:16],
                }
                rows.append(row)
    data.write_json(root / "samples/1.json", {"samples": rows})
    identity = {"dimension": 2, "synthetic": True}

    def synthetic_extract(root, rows):
        (root / "features").mkdir()
        for row in rows:
            feature = [1, 0] if row["speciesId"] == "species-1" else [0, 1]
            np.save(
                root / "features" / f"{classifier.feature_key(row, identity)}.npy",
                np.array(feature, dtype=np.float32),
            )
        return identity

    monkeypatch.setattr(classifier, "extract", synthetic_extract)
    model = classifier.train(root, output, epochs=60)
    assert model["classCount"] == 2
    assert model["evaluation"]["test"]["linearHead"]["top1"] == 1.0
    assert model["status"] == "experimental-offline-not-deployed"
    weights = safetensors.load_file(str(output / "classifier.safetensors"))
    assert weights["weight"].shape == (2, 2)
    assert len((output / "training-manifest.jsonl").read_text().splitlines()) == 60
    assert json.loads((output / "model.json").read_text())["calibrated"] is False
    with pytest.raises(ValueError, match="already exists"):
        classifier.train(root, output)


def test_partial_class_set_requires_explicit_flag(tmp_path):
    pytest.importorskip("torch")
    data.write_json(tmp_path / "species.json", {"species": [species(), species(2)]})
    with pytest.raises(ValueError, match="Need >=2"):
        classifier.train(tmp_path, tmp_path / "model")


def test_partial_model_never_silently_claims_complete_coverage(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    data.write_json(tmp_path / "species.json", {"species": [species(i) for i in range(1, 4)]})
    monkeypatch.setattr(classifier, "validate_samples", lambda *_: None)
    monkeypatch.setattr(classifier, "eligible_species", lambda *_: [species(1), species(2)])
    with pytest.raises(ValueError, match="Only 2/3"):
        classifier.train(tmp_path, tmp_path / "model")


@pytest.mark.parametrize("status", [403, 429])
def test_public_client_stops_without_retry_on_access_or_rate_limit(status, monkeypatch):
    class Response:
        status_code = status

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    client = data.PublicClient()
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(client.session, "get", get)
    with pytest.raises(PermissionError):
        client.get(f"https://{data.PHOTO_HOST}/photos/1/medium.jpg")
    assert len(calls) == 1
    assert calls[0]["allow_redirects"] is False


def test_dataset_lock_prevents_concurrent_writers(tmp_path):
    with data.dataset_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another collector"):
            with data.dataset_lock(tmp_path):
                pytest.fail("Second writer acquired lock.")


def test_user_pause_blocks_collection_and_training_without_removing_data(tmp_path):
    data.write_json(tmp_path / "paused.json", {"reason": "Replace biased shortlist"})
    existing = tmp_path / "existing-image.jpg"
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Dataset paused by user"):
        data.collect(tmp_path, None, None)
    with pytest.raises(ValueError, match="Dataset paused by user"):
        classifier.train(tmp_path, tmp_path / "model")
    assert existing.read_bytes() == b"preserve"
    assert not (tmp_path / "model").exists()
