"""Authenticated private journal and one bounded CPU inference worker."""

import threading

from flask import Blueprint, g, request, send_file

from yizhu_server.auth import auth_required, json_body
from yizhu_server.journal_store import MAX_UPLOAD, JournalStore, fail
from yizhu_server.recognition_candidates import present_recognition
from yizhu_server.recognition_solutions import DEFAULT_SOLUTION, create_provider, get_solution


class RecognitionRuntime:
    def __init__(self, store: JournalStore, provider_factory=None, start=True, *,
                 solution_id=DEFAULT_SOLUTION):
        self.store = store
        self.solution = get_solution(solution_id)
        self.provider_factory = provider_factory
        self.provider = None
        self.state = "warming"
        self.wake = threading.Event()
        self.stop = threading.Event()
        if start:
            threading.Thread(target=self.run, name="yizhu-inference", daemon=True).start()

    def solution_info(self):
        if self.provider_factory is not None:
            return {"id": "custom", "version": None}
        info = self.solution.public_info()
        if self.provider is not None and hasattr(self.provider, "public_model_info"):
            info.update(self.provider.public_model_info())
        return info

    def process_one(self):
        from yizhu_server.recognition import read_photo

        job = self.store.next_job()
        if not job:
            return False
        owner, record = job
        try:
            with read_photo(self.store.photo_path(owner, record["photo"]["id"])) as image:
                result = self.provider.recognize(image)
            self.store.finish(owner, record["id"], result=result)
        except Exception:
            self.store.finish(owner, record["id"], error="识别未完成，请重试或先记录为待确认。")
        return True

    def run(self):
        try:
            if self.provider_factory:
                self.provider = self.provider_factory()
            else:
                self.provider = create_provider(self.solution.id, self.store.catalog)
            self.state = "ready"
        except Exception:
            self.state = "failed"
            while job := self.store.next_job():
                self.store.finish(job[0], job[1]["id"], error="模型暂不可用，请稍后重试。")
            return
        while not self.stop.is_set():
            try:
                if not self.process_one():
                    self.wake.wait(2)
                    self.wake.clear()
            except Exception:
                self.state = "failed"
                return  # Pending rows remain recoverable after a service restart.


def create_journal_blueprint(runtime: RecognitionRuntime):
    api = Blueprint("journal", __name__, url_prefix="/api")
    store = runtime.store

    def catalog_parameters():
        try:
            offset = int(request.args.get("offset", "0"))
            limit = int(request.args.get("limit", "30"))
            chinese_only = request.args.get("chineseOnly", "0")
            if offset < 0 or offset > 200_000 or not 1 <= limit <= 50 or chinese_only not in {"0", "1"}:
                raise ValueError
            return {
                "parent": request.args.get("parent", ""),
                "growth_form": request.args.get("growthForm", ""),
                "offset": offset, "limit": limit,
                "chinese_only": chinese_only == "1",
            }
        except ValueError:
            fail(400, "分页参数无效。")

    @api.get("/catalog")
    def catalog_info():
        return store.catalog.metadata

    @api.get("/catalog/taxa")
    def catalog_taxa():
        if not store.catalog.available:
            fail(503, "植物名录尚未准备好，请稍后重试。")
        try:
            return store.catalog.children(**catalog_parameters())
        except ValueError as error:
            fail(400, str(error))

    @api.get("/catalog/species")
    def catalog_species():
        if not store.catalog.available:
            fail(503, "植物名录尚未准备好，请稍后重试。")
        try:
            return store.catalog.search(q=request.args.get("q", ""), **catalog_parameters())
        except ValueError as error:
            fail(400, str(error))

    @api.get("/catalog/species/<identifier>")
    def catalog_detail(identifier):
        item = store.catalog.get(identifier)
        if item is None:
            fail(404, "名录中没有此物种。")
        return item

    def response(item=None):
        return {
            "snapshot": {**store.snapshot(g.user_id), "engine": runtime.state},
            "item": present_recognition(item) if isinstance(item, dict) else item,
        }

    @api.get("/journal")
    @auth_required
    def snapshot():
        return {**store.snapshot(g.user_id), "engine": runtime.state}

    @api.post("/photos")
    @auth_required
    def upload():
        if set(request.form) != {"requestId"} or set(request.files) != {"file"}:
            fail(400, "请上传单张照片。")
        return store.upload(g.user_id, request.form["requestId"],
                            request.files["file"].read(MAX_UPLOAD + 1))

    @api.get("/photos/content")
    @auth_required
    def photo():
        # Photo IDs, like bearer tokens, never go in URLs/proxy access logs.
        path = store.photo_path(g.user_id, request.headers.get("X-Photo-Id", ""))
        if not path.is_file():
            fail(404, "照片文件已不可用。")
        return send_file(path, mimetype="image/jpeg", conditional=False, max_age=0)

    @api.post("/recognitions")
    @auth_required
    def recognize():
        if runtime.state == "failed":
            fail(503, "模型暂不可用，请稍后重试；不会返回演示结果。")
        item = store.start_recognition(g.user_id, json_body({"photoId", "requestId"}))
        runtime.wake.set()
        return response(item), 202

    @api.post("/recognitions/<identifier>/retry")
    @auth_required
    def retry(identifier):
        json_body(set())
        if runtime.state == "failed":
            fail(503, "模型暂不可用，请稍后重试。")
        item = store.retry(g.user_id, identifier)
        runtime.wake.set()
        return response(item)

    @api.post("/recognitions/<identifier>/confirm")
    @auth_required
    def confirm(identifier):
        data = json_body({"speciesId"})
        if "speciesId" not in data:
            fail(400, "请选择物种或待确认。")
        return response(store.confirm(g.user_id, identifier, data["speciesId"]))

    @api.post("/plants")
    @auth_required
    def create():
        return response(store.create_plant(
            g.user_id, json_body({"recognitionId", "nickname", "location", "story", "description"}),
        ))

    @api.post("/plants/<identifier>/edit")
    @auth_required
    def edit(identifier):
        return response(store.update_plant(
            g.user_id, identifier, json_body({"nickname", "location", "story", "speciesId", "description"}),
        ))

    @api.post("/plants/<identifier>/observations")
    @auth_required
    def append(identifier):
        return response(store.append(
            g.user_id, identifier,
            json_body({"photoId", "requestId", "note", "recognitionId", "confirmedSamePlant"}),
        ))

    @api.post("/observations/<identifier>/edit")
    @auth_required
    def edit_observation(identifier):
        return response(store.update_observation(g.user_id, identifier, json_body({"note"})))

    @api.post("/journal/<kind>/<identifier>/delete")
    @auth_required
    def delete(kind, identifier):
        json_body(set())
        if kind not in {"recognitions", "plants", "observations"}:
            fail(404, "记录类型不存在。")
        store.delete(g.user_id, kind, identifier)
        return response()

    @api.post("/journal/clear")
    @auth_required
    def clear():
        if json_body({"confirm"}).get("confirm") is not True:
            fail(400, "清空数据需再次确认。")
        store.clear(g.user_id)
        return response()

    return api
