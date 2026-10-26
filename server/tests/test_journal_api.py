"""Journal contract tests. Synthetic auth, local generated images, no model or network."""

import io

import pytest

from yizhu_server.app import create_app
from yizhu_server.authclient import AuthError
from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_import import build_catalog
from yizhu_server.journal_api import RecognitionRuntime
from yizhu_server.journal_store import JournalStore
from yizhu_server.recognition import MODEL_REVISION
from yizhu_server.settings import Settings

Image = pytest.importorskip("PIL.Image")


class Auth:
    configured = True

    def userinfo(self, token):
        if token not in {"alice", "bob"}:
            raise AuthError(401, "UNAUTHORIZED", "请登录。")
        return {"sub": token}


class Provider:
    def recognize(self, _image):
        return {
            "source": "BioCLIP", "revision": MODEL_REVISION,
            "candidates": [
                {"label_id": "plant-001", "kind": "plant", "score": 0.9},
                {"label_id": "control-001", "kind": "nonplant", "score": 0.1},
            ],
        }


def headers(owner="alice"):
    return {"Authorization": f"Bearer {owner}"}


@pytest.fixture
def api(tmp_path, national_names):
    path = tmp_path / "catalog"
    build_catalog([
        [["Plantae", "Testophyta", "Testopsida", "Testales", "Testaceae", genus, epithet], ""]
        for genus, epithet in [("Ginkgo", "biloba"), ("Jasminum", "sambac")]
    ], path, chinese_names=national_names)
    store = JournalStore(tmp_path / "private", catalog=PlantCatalog(path))
    runtime = RecognitionRuntime(store, start=False)
    runtime.provider = Provider()
    runtime.state = "ready"
    app = create_app(Settings(h5_dist=tmp_path / "public"), auth_client=Auth(), recognition_runtime=runtime)
    app.config["TESTING"] = True
    return app.test_client(), store, runtime


def upload(client, owner="alice", request_id="photo-one", color="green"):
    image = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(image, "PNG")
    image.seek(0)
    return client.post("/api/photos", headers=headers(owner),
                       data={"file": (image, "photo.png"), "requestId": request_id})


def start(client, photo_id, owner="alice", request_id="rec-one"):
    return client.post("/api/recognitions", headers=headers(owner),
                       json={"photoId": photo_id, "requestId": request_id})


def create(client, recognition_id, nickname="我的植物"):
    return client.post("/api/plants", headers=headers(), json={
        "recognitionId": recognition_id, "nickname": nickname, "location": "", "story": "",
    })


def test_close_candidates_project_old_history_without_rewriting_raw_data(api):
    client, store, runtime = api
    with store.catalog.connect() as db:
        ids = [row["id"] for row in db.execute(
            "SELECT id FROM species WHERE is_visible=1 ORDER BY id LIMIT 2",
        )]
    assert len(ids) == 2
    runtime.provider.recognize = lambda _: {
        "source": "Pl@ntNet", "revision": "synthetic-legacy",
        "score_type": "softmax_not_calibrated_confidence", "status": "requires_review",
        "candidates": [
            {"species_id": identifier, "kind": "plant", "score": score}
            for identifier, score in zip(ids, [.98, .01], strict=True)
        ],
    }
    photo = upload(client).get_json()
    record = start(client, photo["id"]).get_json()["item"]
    assert runtime.process_one()
    with store.transaction() as db:
        before = store.get(db, "alice", "recognitions", record["id"])
    assert len(before["rawResult"]["candidates"]) == 2
    shown = client.get("/api/journal", headers=headers()).get_json()["recognitions"][0]
    assert len(shown["rawResult"]["candidates"]) == 1
    assert shown["rawResult"]["original"] == before["rawResult"]["original"]
    with store.transaction() as db:
        assert store.get(db, "alice", "recognitions", record["id"]) == before
    # Manual correction to a filtered-out catalog entry still works.
    answer = client.post(f"/api/recognitions/{record['id']}/confirm", headers=headers(),
                         json={"speciesId": ids[1]})
    assert answer.status_code == 200
    assert answer.json["item"]["confirmedSpeciesId"] == ids[1]
    assert len(answer.json["item"]["rawResult"]["candidates"]) == 1
    assert not client.get("/api/journal", headers=headers("bob")).json["recognitions"]


def test_real_contract_persists_history_plant_and_observations(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    job = start(client, photo["id"])
    assert job.status_code == 202
    record = job.get_json()["item"]
    assert record["status"] == "running"
    assert start(client, photo["id"]).get_json()["item"]["id"] == record["id"]
    assert runtime.process_one()
    snapshot = client.get("/api/journal", headers=headers()).get_json()
    result = snapshot["recognitions"][0]
    assert result["rawResult"]["isDemo"] is False
    assert result["confirmedSpeciesId"] is None
    assert result["rawResult"]["candidates"][0]["confidence"] is None
    assert result["rawResult"]["original"]["candidates"][1]["kind"] == "nonplant"
    plant = create(client, record["id"]).get_json()["item"]
    assert plant["speciesId"] is None
    assert create(client, record["id"]).get_json()["item"]["id"] == plant["id"]
    changed = client.post(f"/api/plants/{plant['id']}/edit", headers=headers(), json={
        "nickname": "新昵称", "location": "", "story": "用户故事", "speciesId": "bioclip:plant-001",
    }).get_json()["item"]
    assert changed["id"] == plant["id"]
    second = upload(client, request_id="second").get_json()
    payload = {"photoId": second["id"], "note": "新叶", "requestId": "observation-one"}
    response = client.post(f"/api/plants/{plant['id']}/observations", headers=headers(), json=payload)
    observation = response.get_json()["item"]
    assert observation["plantId"] == plant["id"]
    assert client.post(f"/api/plants/{plant['id']}/observations", headers=headers(), json=payload
                       ).get_json()["item"]["id"] == observation["id"]
    client.post(f"/api/observations/{observation['id']}/edit", headers=headers(), json={"note": "更新"})
    restored = JournalStore(store.directory, catalog=store.catalog).snapshot("alice")
    assert len(restored["plants"]) == 1
    assert len(restored["observations"]) == 2
    assert restored["plants"][0]["story"] == "用户故事"


def test_private_ownership_applies_to_every_resource(api):
    client, _, runtime = api
    photo = upload(client).get_json()
    rec = start(client, photo["id"]).get_json()["item"]
    runtime.process_one()
    plant = create(client, rec["id"]).get_json()["item"]
    assert client.get("/api/photos/content", headers={"X-Photo-Id": photo["id"]}).status_code == 401
    assert client.get("/api/photos/content", headers={
        **headers("bob"), "X-Photo-Id": photo["id"],
    }).status_code == 404
    assert client.get("/api/journal", headers=headers("bob")).get_json()["plants"] == []
    assert start(client, photo["id"], owner="bob").status_code == 404
    for path, data in [
        (f"/api/plants/{plant['id']}/edit", {"nickname": "forged"}),
        (f"/api/journal/plants/{plant['id']}/delete", {}),
        (f"/api/recognitions/{rec['id']}/confirm", {"speciesId": None}),
    ]:
        assert client.post(path, headers=headers("bob"), json=data).status_code == 404
    assert client.post("/api/plants", headers=headers(),
                       json={"ownerId": "bob", "recognitionId": rec["id"]}).status_code == 400
    response = client.get("/api/photos/content", headers={**headers(), "X-Photo-Id": photo["id"]})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"


def test_corrections_preserve_original_results_and_link_requires_confirmation(api):
    client, _, runtime = api
    photo = upload(client).get_json()
    first = start(client, photo["id"]).get_json()["item"]
    runtime.process_one()
    plant = create(client, first["id"]).get_json()["item"]
    second = start(client, photo["id"], request_id="rec-two").get_json()["item"]
    runtime.process_one()
    before = client.get("/api/journal", headers=headers()).get_json()["recognitions"][0]["rawResult"]
    client.post(f"/api/recognitions/{second['id']}/confirm", headers=headers(),
                json={"speciesId": "bioclip:plant-002"})
    path = f"/api/plants/{plant['id']}/observations"
    data = {"recognitionId": second["id"], "note": "", "requestId": "link"}
    assert client.post(path, headers=headers(), json=data).status_code == 400
    data["confirmedSamePlant"] = True
    linked = client.post(path, headers=headers(), json=data).get_json()["item"]
    assert client.post(path, headers=headers(), json=data).get_json()["item"]["id"] == linked["id"]
    snapshot = client.get("/api/journal", headers=headers()).get_json()
    assert snapshot["plants"][0]["speciesId"] == "bioclip:plant-002"
    assert snapshot["recognitions"][0]["rawResult"] == before


def test_deletion_is_reference_safe_and_clear_is_owner_scoped(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    bob = upload(client, owner="bob").get_json()
    rec = start(client, photo["id"]).get_json()["item"]
    runtime.process_one()
    plant = create(client, rec["id"]).get_json()["item"]
    # A fresh in-flight upload must not be collected by unrelated deletions.
    unattached = upload(client, request_id="fresh-upload").get_json()
    client.post(f"/api/journal/recognitions/{rec['id']}/delete", headers=headers(), json={})
    assert store.photo_path("alice", photo["id"]).exists()
    assert store.photo_path("alice", unattached["id"]).exists()
    snap = store.snapshot("alice")
    assert snap["observations"][0]["recognitionId"] is None
    client.post(f"/api/journal/plants/{plant['id']}/delete", headers=headers(), json={})
    assert not (store.files / f"{photo['id']}.jpg").exists()
    assert client.post("/api/journal/clear", headers=headers(), json={}).status_code == 400
    client.post("/api/journal/clear", headers=headers(), json={"confirm": True})
    assert not (store.files / f"{unattached['id']}.jpg").exists()
    assert store.photo_path("bob", bob["id"]).exists()


def test_queued_jobs_survive_restart_but_deleted_jobs_never_resurrect(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    record = start(client, photo["id"]).get_json()["item"]
    assert JournalStore(store.directory, catalog=store.catalog).next_job()[1]["id"] == record["id"]
    client.post("/api/journal/clear", headers=headers(), json={"confirm": True})
    store.finish("alice", record["id"], result=Provider().recognize(None))
    assert store.snapshot("alice")["recognitions"] == []
    assert runtime.process_one() is False


def test_upload_limits_idempotency_and_model_failure(api):
    client, _, runtime = api
    photo = upload(client).get_json()
    assert upload(client).get_json()["id"] == photo["id"]
    assert upload(client, color="red").status_code == 409
    bad = client.post("/api/photos", headers=headers(),
                      data={"file": (io.BytesIO(b"bad"), "bad.jpg"), "requestId": "bad"})
    assert bad.status_code == 400
    too_large = client.post("/api/photos", headers=headers(), data={
        "file": (io.BytesIO(b"x" * (1024 * 1024 + 1)), "large.jpg"), "requestId": "large",
    })
    assert too_large.status_code == 413
    runtime.state = "failed"
    assert start(client, photo["id"]).status_code == 503


def test_job_limit_and_daily_quota_survive_clearing_history(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    rec = start(client, photo["id"]).get_json()["item"]
    assert start(client, photo["id"], request_id="another").status_code == 429
    runtime.process_one()
    with store.transaction() as db:
        db.execute("UPDATE usage SET count=30 WHERE owner='alice'")
    client.post(f"/api/journal/recognitions/{rec['id']}/delete", headers=headers(), json={})
    another = upload(client, request_id="after-clear").get_json()
    assert start(client, another["id"], request_id="over-quota").status_code == 429


def test_private_data_path_cannot_be_served_as_h5(tmp_path):
    with pytest.raises(ValueError, match="Private"):
        create_app(Settings(h5_dist=tmp_path, data_dir=tmp_path / "private"))


def test_private_description_is_persistent_owned_editable_and_not_species_knowledge(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    rec = start(client, photo["id"]).get_json()["item"]
    runtime.process_one()
    response = client.post("/api/plants", headers=headers(), json={
        "recognitionId": rec["id"], "nickname": "窗边植物", "story": "这是我的故事",
        "description": "叶片有浅色斑纹。\n靠窗的一侧更茂盛。",
    })
    assert response.status_code == 200
    plant = response.get_json()["item"]
    assert plant["description"].startswith("叶片")
    assert plant["story"] == "这是我的故事"
    assert JournalStore(store.directory, catalog=store.catalog).snapshot("alice")["plants"][0]["description"] == plant["description"]
    path = f"/api/plants/{plant['id']}/edit"
    old_client = {"nickname": "换个名字", "location": "", "story": "故事保留", "speciesId": None}
    assert client.post(path, headers=headers(), json=old_client).get_json()["item"]["description"] == plant["description"]
    assert client.post(path, headers=headers("bob"), json={**old_client, "description": "越权"}).status_code == 404
    for invalid in [None, {}, "x" * 1001]:
        assert client.post(path, headers=headers(), json={**old_client, "description": invalid}).status_code == 400
    assert store.snapshot("alice")["plants"][0]["description"] == plant["description"]
    edited = client.post(path, headers=headers(), json={**old_client, "description": "新描述"}).get_json()["item"]
    assert edited["id"] == plant["id"] and edited["description"] == "新描述"
    cleared = client.post(path, headers=headers(), json={**old_client, "description": ""}).get_json()["item"]
    assert cleared["description"] == ""
    assert client.post(path, headers=headers(), json={**old_client, "knowledge": {"summary": "伪造"}}).status_code == 400


def finished_record(client, runtime, photo_id, request_id):
    record = start(client, photo_id, request_id=request_id).get_json()["item"]
    assert runtime.process_one()
    return record


def link(client, plant_id, rec_id, owner="alice"):
    return client.post(f"/api/plants/{plant_id}/observations", headers=headers(owner), json={
        "recognitionId": rec_id, "requestId": f"link:{rec_id}",
        "note": "", "confirmedSamePlant": True,
    })


@pytest.mark.parametrize("association", ["create", "link"])
def test_confirmation_syncs_linked_plant_without_changing_identity_or_raw_results(api, association):
    client, store, runtime = api
    photo = upload(client).get_json()
    rec = finished_record(client, runtime, photo["id"], "source")
    plant = create(client, rec["id"]).get_json()["item"]
    if association == "link":
        rec = finished_record(client, runtime, photo["id"], "linked")
        assert link(client, plant["id"], rec["id"]).status_code == 200
    before = store.snapshot("alice")
    original = next(item for item in before["recognitions"] if item["id"] == rec["id"])["rawResult"]
    for species_id in ["bioclip:plant-001", "bioclip:plant-002", None]:
        response = client.post(f"/api/recognitions/{rec['id']}/confirm", headers=headers(),
                               json={"speciesId": species_id})
        assert response.status_code == 200
        snapshot = response.get_json()["snapshot"]
        updated = snapshot["plants"][0]
        assert updated == {**plant, "speciesId": species_id}
        updated_rec = next(item for item in snapshot["recognitions"] if item["id"] == rec["id"])
        assert updated_rec["confirmedSpeciesId"] == species_id
        assert updated_rec["linkedPlantId"] == plant["id"]
        assert updated_rec["rawResult"] == original
    assert JournalStore(store.directory, catalog=store.catalog).snapshot("alice")["plants"][0]["speciesId"] is None


def test_recognition_can_only_belong_to_one_plant_across_create_and_link(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    first = finished_record(client, runtime, photo["id"], "first")
    first_plant = create(client, first["id"]).get_json()["item"]
    second = finished_record(client, runtime, photo["id"], "second")
    second_plant = create(client, second["id"]).get_json()["item"]
    assert link(client, second_plant["id"], first["id"]).status_code == 409
    third = finished_record(client, runtime, photo["id"], "third")
    linked = link(client, first_plant["id"], third["id"])
    assert linked.status_code == 200
    assert link(client, first_plant["id"], third["id"]).get_json()["item"] == linked.get_json()["item"]
    assert link(client, second_plant["id"], third["id"]).status_code == 409
    # A create retry after linking returns the same plant, never a new plant.
    assert create(client, third["id"]).get_json()["item"]["id"] == first_plant["id"]
    assert len(store.snapshot("alice")["plants"]) == 2
    assert len(store.snapshot("alice")["observations"]) == 3
    assert link(client, first_plant["id"], third["id"], owner="bob").status_code == 404
    bob_photo = upload(client, owner="bob").get_json()
    bob_rec = start(client, bob_photo["id"], owner="bob").get_json()["item"]
    runtime.process_one()
    assert link(client, first_plant["id"], bob_rec["id"]).status_code == 404


@pytest.mark.parametrize("legacy", [False, True])
def test_link_survives_observation_deletion_restart_and_releases_on_plant_deletion(api, legacy):
    client, store, runtime = api
    photo = upload(client).get_json()
    source = finished_record(client, runtime, photo["id"], "source")
    plant = create(client, source["id"]).get_json()["item"]
    rec = finished_record(client, runtime, photo["id"], "linked")
    obs = link(client, plant["id"], rec["id"]).get_json()["item"]
    if legacy:
        with store.transaction() as db:
            item = store.get(db, "alice", "recognitions", rec["id"])
            item.pop("linkedPlantId")
            item.pop("associationConflict")
            store.put(db, "alice", "recognitions", item)
    assert client.post(f"/api/journal/observations/{obs['id']}/delete",
                       headers=headers(), json={}).status_code == 200
    restored = JournalStore(store.directory, catalog=store.catalog).snapshot("alice")
    restored_rec = next(item for item in restored["recognitions"] if item["id"] == rec["id"])
    assert restored_rec["linkedPlantId"] == plant["id"]
    assert create(client, rec["id"]).get_json()["item"]["id"] == plant["id"]
    client.post(f"/api/recognitions/{rec['id']}/confirm", headers=headers(),
                json={"speciesId": "bioclip:plant-002"})
    assert store.snapshot("alice")["plants"][0]["speciesId"] == "bioclip:plant-002"
    client.post(f"/api/journal/plants/{plant['id']}/delete", headers=headers(), json={})
    assert all(item["linkedPlantId"] is None for item in store.snapshot("alice")["recognitions"])
    recreated = create(client, rec["id"]).get_json()["item"]
    assert recreated["id"] != plant["id"]
    assert recreated["speciesId"] == "bioclip:plant-002"


def test_legacy_source_links_backfill_and_conflicting_links_are_not_silently_merged(api):
    client, store, runtime = api
    photo = upload(client).get_json()
    first = finished_record(client, runtime, photo["id"], "first")
    plant = create(client, first["id"]).get_json()["item"]
    second = finished_record(client, runtime, photo["id"], "second")
    other = create(client, second["id"]).get_json()["item"]
    with store.transaction() as db:
        rec = store.get(db, "alice", "recognitions", first["id"])
        rec.pop("linkedPlantId")
        rec.pop("associationConflict")
        store.put(db, "alice", "recognitions", rec)
        for obs in store.docs(db, "alice", "observations"):
            if obs["recognitionId"] == first["id"]:
                db.execute("DELETE FROM docs WHERE id=?", (obs["id"],))
    snap = store.snapshot("alice")
    assert next(item for item in snap["recognitions"]
                if item["id"] == first["id"])["linkedPlantId"] == plant["id"]
    with store.transaction() as db:
        store.observation(db, "alice", other["id"], photo, "旧版本重复关联", first["id"])
    snap = store.snapshot("alice")
    rec = next(item for item in snap["recognitions"] if item["id"] == first["id"])
    assert rec["associationConflict"] is True
    assert create(client, first["id"]).status_code == 409
    assert link(client, other["id"], first["id"]).status_code == 409
    assert client.post(f"/api/recognitions/{first['id']}/confirm", headers=headers(),
                       json={"speciesId": "bioclip:plant-002"}).status_code == 409
    assert store.snapshot("alice") == snap  # No silent data deletion or species change.
    client.post(f"/api/journal/plants/{other['id']}/delete", headers=headers(), json={})
    rec = next(item for item in store.snapshot("alice")["recognitions"] if item["id"] == first["id"])
    assert rec["associationConflict"] is False
    assert rec["linkedPlantId"] == plant["id"]


def test_concurrent_create_and_link_cannot_assign_two_plants(api):
    from concurrent.futures import ThreadPoolExecutor

    client, store, runtime = api
    photo = upload(client).get_json()
    first = finished_record(client, runtime, photo["id"], "first")
    plant = create(client, first["id"]).get_json()["item"]
    rec = finished_record(client, runtime, photo["id"], "race")
    another_store = JournalStore(store.directory, catalog=store.catalog)

    def create_or_link(is_create):
        try:
            if is_create:
                return store.create_plant("alice", {"recognitionId": rec["id"], "nickname": "新植物"})
            return another_store.append("alice", plant["id"], {
                "recognitionId": rec["id"], "confirmedSamePlant": True, "requestId": "race",
            })
        except AuthError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(create_or_link, [True, False]))
    snap = store.snapshot("alice")
    linked_obs = [item for item in snap["observations"] if item["recognitionId"] == rec["id"]]
    assert len(linked_obs) == 1
    updated = next(item for item in snap["recognitions"] if item["id"] == rec["id"])
    assert updated["linkedPlantId"] == linked_obs[0]["plantId"]
    assert updated["associationConflict"] is False
