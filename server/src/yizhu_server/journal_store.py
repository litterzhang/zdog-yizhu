"""Single-host private journal persistence. All lookups are scoped to a verified subject."""

import hashlib
import io
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock

from yizhu_server.authclient import AuthError
from yizhu_server.catalog import PlantCatalog
from yizhu_server.recognition import read_photo
from yizhu_server.recognition_candidates import present_recognition
from yizhu_server.species_presentation import journal_species

MAX_UPLOAD = 900 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid.uuid4())


def fail(status, message):
    raise AuthError(status, "JOURNAL_ERROR", message)


def key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:._-]{1,160}", value):
        fail(400, "请求标识无效，请重新操作。")
    return value


def text(data, name, limit, required=False):
    value = data.get(name, "")
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        fail(400, f"{name} 不能为空或超过 {limit} 字。")
    return value.strip()


class JournalStore:
    def __init__(self, directory: Path, catalog=None):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        self.directory = directory
        self.files = directory / "photos"
        self.files.mkdir(exist_ok=True, mode=0o700)
        self.lock = RLock()
        self.pending_unlinks = []
        self.database = directory / "journal.sqlite3"
        self.catalog = catalog or PlantCatalog()
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS docs (
                    kind TEXT NOT NULL, id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    data TEXT NOT NULL, created TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS docs_owner_kind ON docs(owner,kind,created);
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_key TEXT NOT NULL,
                    digest TEXT NOT NULL, created REAL NOT NULL, UNIQUE(owner,request_key));
                CREATE INDEX IF NOT EXISTS photos_owner ON photos(owner);
                CREATE TABLE IF NOT EXISTS usage (
                    owner TEXT NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL,
                    PRIMARY KEY(owner,day));
            """)
        os.chmod(self.database, 0o600)
        self.sweep_orphans()

    @contextmanager
    def transaction(self):
        with self.lock:
            db = sqlite3.connect(self.database, timeout=15)
            db.row_factory = sqlite3.Row
            pending = []
            self.pending_unlinks = pending
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
                for path in pending:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass  # Startup sweep retries physical deletion.
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def docs(self, db, owner, kind):
        return [json.loads(row["data"]) for row in db.execute(
            "SELECT data FROM docs WHERE owner=? AND kind=? ORDER BY created DESC", (owner, kind),
        )]

    def get(self, db, owner, kind, identifier):
        if not isinstance(identifier, str):
            fail(400, "记录标识无效。")
        row = db.execute(
            "SELECT data FROM docs WHERE owner=? AND kind=? AND id=?", (owner, kind, identifier),
        ).fetchone()
        if not row:
            fail(404, "记录不存在或无权访问。")
        return json.loads(row["data"])

    def put(self, db, owner, kind, document):
        db.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            (kind, document["id"], owner, json.dumps(document, ensure_ascii=False),
             document.get("createdAt", document.get("observedAt", now()))),
        )

    def photo(self, db, owner, identifier):
        if not isinstance(identifier, str):
            fail(400, "照片标识无效。")
        row = db.execute("SELECT id FROM photos WHERE owner=? AND id=?", (owner, identifier)).fetchone()
        if not row:
            fail(404, "照片不存在或无权访问。")
        return {"id": row["id"], "uri": "", "origin": "remote"}

    def photo_path(self, owner, identifier):
        with self.transaction() as db:
            self.photo(db, owner, identifier)
        return self.files / f"{identifier}.jpg"

    def snapshot(self, owner):
        with self.transaction() as db:
            self.resolve_links(db, owner)
            snapshot = {kind: self.docs(db, owner, kind) for kind in (
                "recognitions", "plants", "observations",
            )}
            snapshot["recognitions"] = [
                present_recognition(record) for record in snapshot["recognitions"]
            ]
            identifiers = {plant["speciesId"] for plant in snapshot["plants"] if plant["speciesId"]}
            for rec in snapshot["recognitions"]:
                if rec["confirmedSpeciesId"]:
                    identifiers.add(rec["confirmedSpeciesId"])
                identifiers.update(row["speciesId"] for row in (rec.get("rawResult") or {}).get("candidates", []))
            return {**snapshot, "species": [
                        journal_species(item) for item in self.catalog.selected(identifiers)
                    ],
                    "catalog": self.catalog.metadata}

    def resolve_link(self, db, owner, rec, plants=None, observations=None):
        """Backfill legacy links without merging or deleting any user records."""
        if plants is None:
            plants = self.docs(db, owner, "plants")
        if observations is None:
            observations = self.docs(db, owner, "observations")
        valid_ids = {plant["id"] for plant in plants}
        linked_ids = {
            obs["plantId"] for obs in observations
            if obs.get("recognitionId") == rec["id"]
        }
        linked_ids.update(plant["id"] for plant in plants
                          if plant.get("_sourceRecognitionId") == rec["id"])
        if rec.get("linkedPlantId"):
            linked_ids.add(rec["linkedPlantId"])
        linked_ids &= valid_ids
        conflict = len(linked_ids) > 1
        linked_id = next(iter(linked_ids)) if len(linked_ids) == 1 else None
        # Retain a durable link even when its individual observation is deleted.
        if conflict:
            linked_id = rec.get("linkedPlantId")
        if rec.get("linkedPlantId") != linked_id or rec.get("associationConflict") != conflict:
            rec.update(linkedPlantId=linked_id, associationConflict=conflict)
            self.put(db, owner, "recognitions", rec)
        return linked_id

    def resolve_links(self, db, owner):
        plants = self.docs(db, owner, "plants")
        observations = self.docs(db, owner, "observations")
        for rec in self.docs(db, owner, "recognitions"):
            self.resolve_link(db, owner, rec, plants, observations)

    def linked_plant(self, db, owner, rec):
        linked_id = self.resolve_link(db, owner, rec)
        if rec.get("associationConflict"):
            fail(409, "这条旧足迹关联了多个株记，请先整理重复关联，再确认物种或建档。")
        return self.get(db, owner, "plants", linked_id) if linked_id else None

    def upload(self, owner, request_key, raw):
        key(request_key)
        if not raw or len(raw) > MAX_UPLOAD:
            fail(413, "照片需小于 900 KiB，请压缩后重试。")
        temporary = self.files / f"{uid()}.upload"
        try:
            with temporary.open("xb") as stream:
                os.chmod(temporary, 0o600)
                stream.write(raw)
            try:
                with read_photo(temporary) as image:
                    image.thumbnail((1600, 1600))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=85, optimize=True)
                    normalized = buffer.getvalue()
            except ValueError:
                fail(400, "照片无法读取。请使用不超过 20MP 的静态 JPEG、PNG 或 WebP。")
            digest = hashlib.sha256(raw).hexdigest()
            with self.transaction() as db:
                previous = db.execute(
                    "SELECT id,digest FROM photos WHERE owner=? AND request_key=?",
                    (owner, request_key),
                ).fetchone()
                if previous:
                    if previous["digest"] != digest:
                        fail(409, "重复请求的照片不同，请重新选择照片。")
                    return self.photo(db, owner, previous["id"])
                self.collect(db, owner, old_only=True)
                if db.execute("SELECT count(*) FROM photos WHERE owner=?", (owner,)).fetchone()[0] >= 200:
                    fail(429, "每个账号最多保存 200 张照片，请先删除不需要的记录。")
                if db.execute("SELECT count(*) FROM photos").fetchone()[0] >= 2000:
                    fail(503, "存储额度已满，请稍后再试。")
                identifier = uid()
                target = self.files / f"{identifier}.jpg"
                with target.open("xb") as stream:
                    os.chmod(target, 0o600)
                    stream.write(normalized)
                db.execute("INSERT INTO photos VALUES (?,?,?,?,?)",
                           (identifier, owner, request_key, digest, time.time()))
                return self.photo(db, owner, identifier)
        finally:
            temporary.unlink(missing_ok=True)

    def require_species(self, value, *, previous=None):
        if value is not None and (
            not isinstance(value, str) or self.catalog.get(value, include_hidden=value == previous) is None
        ):
            fail(400, "物种不在当前名录中，可先记录为待确认。")
        return value

    def quota(self, db, owner):
        active = db.execute(
            "SELECT owner,data FROM docs WHERE kind='recognitions' "
            "AND json_extract(data,'$.status')='running'",
        ).fetchall()
        if len(active) >= 8 or any(row["owner"] == owner for row in active):
            fail(429, "已有识别任务在等待，请先到足迹查看，稍后再试。")
        day = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        db.execute("DELETE FROM usage WHERE day < ?", (day,))
        row = db.execute("SELECT count FROM usage WHERE owner=? AND day=?", (owner, day)).fetchone()
        if row and row[0] >= 30:
            fail(429, "已达到 每日 30 次识别额度，明天再试。")
        db.execute(
            "INSERT INTO usage VALUES (?,?,1) ON CONFLICT(owner,day) DO UPDATE SET count=count+1",
            (owner, day),
        )

    def start_recognition(self, owner, data):
        request_key = key(data.get("requestId"))
        with self.transaction() as db:
            for item in self.docs(db, owner, "recognitions"):
                if item["_requestId"] == request_key:
                    if item["photo"]["id"] != data.get("photoId"):
                        fail(409, "重复请求内容不同。")
                    return item
            photo = self.photo(db, owner, data.get("photoId"))
            if len(self.docs(db, owner, "recognitions")) >= 1000:
                fail(429, "足迹数量已满，请删除不需要的足迹。")
            self.quota(db, owner)
            item = {"id": uid(), "ownerId": owner, "photo": photo, "status": "running",
                    "rawResult": None, "confirmedSpeciesId": None, "createdAt": now(),
                    "failure": None, "_requestId": request_key,
                    "linkedPlantId": None, "associationConflict": False}
            self.put(db, owner, "recognitions", item)
            return item

    def retry(self, owner, identifier):
        with self.transaction() as db:
            item = self.get(db, owner, "recognitions", identifier)
            if item["status"] == "failed":
                self.quota(db, owner)
                item.update(status="running", failure=None)
                self.put(db, owner, "recognitions", item)
            return item

    def confirm(self, owner, identifier, species_id):
        with self.transaction() as db:
            item = self.get(db, owner, "recognitions", identifier)
            if item["status"] == "running":
                fail(409, "请等待识别完成。")
            plant = self.linked_plant(db, owner, item)
            item["confirmedSpeciesId"] = self.require_species(species_id, previous=item["confirmedSpeciesId"])
            self.put(db, owner, "recognitions", item)
            if plant:
                plant["speciesId"] = item["confirmedSpeciesId"]
                self.put(db, owner, "plants", plant)
            return item

    def observation(self, db, owner, plant_id, photo, note, rec=None, request_key=None):
        item = {"id": uid(), "ownerId": owner, "plantId": plant_id, "photo": photo,
                "note": note, "observedAt": now(), "recognitionId": rec, "_requestId": request_key}
        self.put(db, owner, "observations", item)
        return item

    def create_plant(self, owner, data):
        with self.transaction() as db:
            rec = self.get(db, owner, "recognitions", data.get("recognitionId"))
            if rec["status"] == "running":
                fail(409, "请等待识别完成。")
            linked = self.linked_plant(db, owner, rec)
            if linked:
                return linked
            if len(self.docs(db, owner, "plants")) >= 200:
                fail(429, "植物档案数量已满。")
            item = {"id": uid(), "ownerId": owner,
                    "nickname": text(data, "nickname", 40, True),
                    "locationDescription": text(data, "location", 80),
                    "story": text(data, "story", 500), "speciesId": rec["confirmedSpeciesId"],
                    "description": text(data, "description", 1000),
                    "coordinates": None, "cover": rec["photo"], "createdAt": now(),
                    "_sourceRecognitionId": rec["id"]}
            self.put(db, owner, "plants", item)
            rec["linkedPlantId"] = item["id"]
            self.put(db, owner, "recognitions", rec)
            self.observation(db, owner, item["id"], rec["photo"], "第一次记住这一株。", rec["id"])
            return item

    def update_plant(self, owner, identifier, data):
        with self.transaction() as db:
            item = self.get(db, owner, "plants", identifier)
            item.update(
                nickname=text(data, "nickname", 40, True),
                locationDescription=text(data, "location", 80),
                story=text(data, "story", 500),
                speciesId=self.require_species(data.get("speciesId"), previous=item.get("speciesId")),
            )
            # Old clients omit description; an unrelated edit must not erase it.
            if "description" in data:
                item["description"] = text(data, "description", 1000)
            self.put(db, owner, "plants", item)
            return item

    def append(self, owner, identifier, data):
        with self.transaction() as db:
            plant = self.get(db, owner, "plants", identifier)
            rec_id = data.get("recognitionId")
            request_key = key(data.get("requestId"))
            existing = self.docs(db, owner, "observations")
            if rec_id:
                if data.get("confirmedSamePlant") is not True:
                    fail(400, "关联个体必须由你确认。")
                rec = self.get(db, owner, "recognitions", rec_id)
                if rec["status"] == "running":
                    fail(409, "请等待识别完成。")
                linked = self.linked_plant(db, owner, rec)
                if linked and linked["id"] != identifier:
                    fail(409, "这条足迹已记入其他株记，请查看原株记。")
                photo = rec["photo"]
            else:
                photo = self.photo(db, owner, data.get("photoId"))
            note = text(data, "note", 500)
            for item in existing:
                if item["plantId"] == identifier and (
                    item["_requestId"] == request_key or (rec_id and item["recognitionId"] == rec_id)
                ):
                    if item["photo"]["id"] != photo["id"] or (
                        item["_requestId"] == request_key and item["note"] != note
                    ):
                        fail(409, "重复请求的照片不同。")
                    return item
            if len(existing) >= 1000:
                fail(429, "观察记录额度已满。")
            if rec_id:
                rec["linkedPlantId"] = identifier
                self.put(db, owner, "recognitions", rec)
                if rec["confirmedSpeciesId"] is not None:
                    plant["speciesId"] = rec["confirmedSpeciesId"]
                    self.put(db, owner, "plants", plant)
            return self.observation(db, owner, identifier, photo, note, rec_id, request_key)

    def update_observation(self, owner, identifier, data):
        with self.transaction() as db:
            item = self.get(db, owner, "observations", identifier)
            item["note"] = text(data, "note", 500)
            self.put(db, owner, "observations", item)
            return item

    def delete(self, owner, kind, identifier):
        with self.transaction() as db:
            item = self.get(db, owner, kind, identifier)
            self.resolve_links(db, owner)
            previously_used = {
                value["cover" if category == "plants" else "photo"]["id"]
                for category in ("plants", "recognitions", "observations")
                for value in self.docs(db, owner, category)
            }
            db.execute("DELETE FROM docs WHERE owner=? AND kind=? AND id=?", (owner, kind, identifier))
            if kind == "plants":
                for rec in self.docs(db, owner, "recognitions"):
                    if rec.get("linkedPlantId") == identifier:
                        rec["linkedPlantId"] = None
                        self.put(db, owner, "recognitions", rec)
                for obs in self.docs(db, owner, "observations"):
                    if obs["plantId"] == identifier:
                        db.execute("DELETE FROM docs WHERE owner=? AND id=?", (owner, obs["id"]))
            if kind == "recognitions":
                for obs in self.docs(db, owner, "observations"):
                    if obs["recognitionId"] == identifier:
                        obs["recognitionId"] = None
                        self.put(db, owner, "observations", obs)
            if kind == "observations":
                plant = self.get(db, owner, "plants", item["plantId"])
                others = [obs for obs in self.docs(db, owner, "observations")
                          if obs["plantId"] == plant["id"]]
                if others and plant["cover"]["id"] == item["photo"]["id"]:
                    plant["cover"] = others[0]["photo"]
                    self.put(db, owner, "plants", plant)
            self.collect(db, owner, targets=previously_used)

    def clear(self, owner):
        with self.transaction() as db:
            db.execute("DELETE FROM docs WHERE owner=?", (owner,))
            self.collect(db, owner)

    def collect(self, db, owner, old_only=False, targets=None):
        used = set()
        for kind in ("recognitions", "plants", "observations"):
            for item in self.docs(db, owner, kind):
                used.add(item["cover" if kind == "plants" else "photo"]["id"])
        for photo in db.execute("SELECT id,created FROM photos WHERE owner=?", (owner,)).fetchall():
            if targets is not None and photo["id"] not in targets:
                continue
            if photo["id"] not in used and (not old_only or photo["created"] < time.time() - 86400):
                # Files are removed only after the enclosing DB transaction commits.
                db.execute("DELETE FROM photos WHERE owner=? AND id=?", (owner, photo["id"]))
                self.pending_unlinks.append(self.files / f"{photo['id']}.jpg")

    def sweep_orphans(self):
        with self.transaction() as db:
            for owner in db.execute("SELECT DISTINCT owner FROM photos").fetchall():
                self.collect(db, owner[0], old_only=True)
            keep = {row[0] for row in db.execute("SELECT id FROM photos")}
            for path in self.files.iterdir():
                if path.suffix == ".jpg" and path.stem not in keep:
                    self.pending_unlinks.append(path)
                elif path.suffix == ".upload" and path.stat().st_mtime < time.time() - 3600:
                    self.pending_unlinks.append(path)

    def next_job(self):
        with self.transaction() as db:
            row = db.execute(
                "SELECT owner,data FROM docs WHERE kind='recognitions' "
                "AND json_extract(data,'$.status')='running' ORDER BY created LIMIT 1",
            ).fetchone()
            return (row["owner"], json.loads(row["data"])) if row else None

    def finish(self, owner, identifier, result=None, error=None):
        with self.transaction() as db:
            try:
                item = self.get(db, owner, "recognitions", identifier)
            except AuthError:
                return  # Deleted while queued/running: never resurrect.
            if item["status"] != "running":
                return
            if error:
                item.update(status="failed", failure=error)
            else:
                plant_candidates = [
                    {"speciesId": row.get("species_id") or f"bioclip:{row['label_id']}", "confidence": None,
                     "score": row["score"]}
                    for row in result["candidates"] if row["kind"] == "plant"
                ]
                visible_candidates = [
                    row for row in plant_candidates if self.catalog.get(row["speciesId"]) is not None
                ]
                hidden_leading = bool(
                    plant_candidates and self.catalog.get(plant_candidates[0]["speciesId"]) is None
                )
                # Preserve all original scores and controls independently of user corrections.
                item["rawResult"] = {
                    "source": result["source"], "version": result["revision"], "isDemo": False,
                    "candidates": visible_candidates, "original": result,
                    "catalogVersion": result.get("label_version"),
                    "scoreType": result.get("score_type"),
                    "reason": result.get("reason") or ("hidden_leading_candidate" if hidden_leading else None),
                }
                item.update(status="succeeded" if visible_candidates and not hidden_leading
                            and result.get("status") != "uncertain"
                            else "uncertain", failure=None)
            self.put(db, owner, "recognitions", item)
