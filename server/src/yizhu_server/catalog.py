"""Read-only, paginated plant taxonomy. Never load the whole flora into a journal."""

import hashlib
import json
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from yizhu_server.recognition import load_labels

CATALOG_REVISION = "91debffb7146c32c89d76feb1eb575b555e2ecc7"
CATALOG_SOURCE = "Imageomics / TreeOfLife-10M"
CATALOG_URL = f"https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/{CATALOG_REVISION}"
CATALOG_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache/catalog/treeoflife-10m"
CATALOG_PATH = CATALOG_DIRECTORY / "catalog.sqlite3"
PLANT_KINGDOMS = {"Plantae", "Viridiplantae"}
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
GROWTH_FORMS = ("herb", "shrub", "tree", "climber", "unknown")
ASSETS = {
    "txt_emb_species.json": (65_731_052, "844e6fabc06cac072214d566b78f40825b154efa9479eb11285030ca038b2ece"),
    "txt_emb_species.npy": (787_435_648, "91ce02dff2433222e3138b8bf7eefa1dd74b30f4d406c16cd3301f66d65ab4ed"),
}


def stable_id(prefix, value):
    return f"{prefix}:{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def verify_asset(path, name):
    size, expected = ASSETS[name]
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing or incomplete catalog asset: {name}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Catalog checksum mismatch: {name}")


def legacy_species():
    """ID compatibility for old private references, not a name source."""
    _, labels = load_labels()
    return {
        f"bioclip:{label.id}": {
            "id": f"bioclip:{label.id}", "commonName": label.name,
            "scientificName": label.name, "taxonomy": None, "knowledge": None,
            "chineseName": None, "chineseAliases": [], "chineseNameOrigin": None,
            "nameSources": [], "isVisible": False,
            "growthForms": [], "sources": [],
        } for label in labels if label.kind == "plant"
    }


class PlantCatalog:
    def __init__(self, path=CATALOG_PATH):
        self.path = Path(path) if path is not None else None
        self.legacy = legacy_species()
        self.available = self.path is not None and self.path.is_file()
        self.metadata = {
            "available": False, "source": CATALOG_SOURCE, "version": CATALOG_REVISION,
            "plantCount": 0, "labelCount": 0, "growthFormCount": 0,
        }
        if self.available:
            with self.connect() as db:
                self.metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
            if self.metadata.get("version") != CATALOG_REVISION:
                raise ValueError("Unexpected catalog revision. Rebuild from the pinned source.")

    def connect(self):
        if not self.available:
            raise ValueError("植物名录尚未安装，请稍后重试。")
        db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return closing(db)

    @lru_cache(maxsize=1024)
    def get(self, identifier, *, include_hidden=False):
        if not isinstance(identifier, str):
            return None
        if self.available:
            with self.connect() as db:
                row = db.execute("SELECT data,is_visible FROM species WHERE id=?", (identifier,)).fetchone()
                if row:
                    if not include_hidden and not row["is_visible"]:
                        return None
                    return json.loads(row["data"])
        item = self.legacy.get(identifier)
        return item if item and (include_hidden or item["isVisible"]) else None

    def selected(self, identifiers):
        """Resolve private historical references, including now-hidden species."""
        identifiers = sorted(identifiers)
        found = {key: self.legacy[key] for key in identifiers if key in self.legacy}
        if self.available:
            with self.connect() as db:
                for start in range(0, len(identifiers), 500):
                    chunk = identifiers[start:start + 500]
                    for row in db.execute(
                        f"SELECT id,data FROM species WHERE id IN ({','.join('?' for _ in chunk)})",
                        chunk,
                    ):
                        found[row["id"]] = json.loads(row["data"])
        return [found[key] for key in identifiers if key in found]

    @staticmethod
    def filters(q="", parent="", growth_form="", chinese_only=False):
        if not isinstance(q, str) or len(q) > 80:
            raise ValueError("搜索内容最多 80 字。")
        if not isinstance(parent, str) or len(parent) > 100:
            raise ValueError("分类标识无效。")
        if growth_form and growth_form not in GROWTH_FORMS:
            raise ValueError("生长形态无效。")
        conditions, params = ["s.is_visible=1"], []
        if chinese_only:
            conditions.append("s.has_chinese_name=1")
        if q.strip():
            # Escape wildcard characters; input is always a bound SQL value.
            value = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("s.search_text LIKE ? ESCAPE '\\'")
            params.append(f"%{value}%")
        if parent:
            conditions.append("EXISTS (SELECT 1 FROM membership m WHERE m.species_id=s.id AND m.taxon_id=?)")
            params.append(parent)
        if growth_form:
            conditions.append("s.growth_form=?")
            params.append(growth_form)
        return (" AND ".join(conditions) or "1=1"), params

    @lru_cache(maxsize=128)
    def search(self, *, q="", parent="", growth_form="", offset=0, limit=30, chinese_only=False):
        if chinese_only and "chineseNameCount" not in self.metadata:
            raise ValueError("中文名索引尚未安装，请更新名录数据或切换全部参考。")
        where, values = self.filters(q, parent, growth_form, chinese_only)
        if offset < 0 or offset > 200_000 or not 1 <= limit <= 50:
            raise ValueError("分页参数无效。")
        with self.connect() as db:
            source = "species s"
            if self.metadata.get("storageSchemaVersion") == 2:
                source = "species_core s JOIN species_zh z ON z.species_id=s.id"
                for field in ("is_visible", "has_chinese_name", "search_text"):
                    where = where.replace(f"s.{field}", f"z.{field}")
            total = db.execute(f"SELECT count(*) FROM {source} WHERE {where}", values).fetchone()[0]
            rows = db.execute(
                "SELECT data FROM species s WHERE s.id IN "
                f"(SELECT s.id FROM {source} WHERE {where} "
                "ORDER BY s.scientific_name,s.id LIMIT ? OFFSET ?) "
                "ORDER BY s.scientific_name,s.id", [*values, limit, offset],
            ).fetchall()
        return {"items": [json.loads(row["data"]) for row in rows], "total": total,
                "offset": offset, "limit": limit}

    @lru_cache(maxsize=128)
    def children(self, *, parent="", growth_form="", offset=0, limit=30, chinese_only=False):
        has_names = "chineseNameCount" in self.metadata
        if chinese_only and not has_names:
            raise ValueError("中文名索引尚未安装，请更新名录数据或切换全部参考。")
        self.filters(parent=parent, growth_form=growth_form)
        if offset < 0 or offset > 200_000 or not 1 <= limit <= 50:
            raise ValueError("分页参数无效。")
        with self.connect() as db:
            if parent and not db.execute("SELECT 1 FROM taxa WHERE id=?", (parent,)).fetchone():
                raise ValueError("分类节点不存在。")
            values = [parent]
            extra, join = "AND s.is_visible=1", "JOIN species s ON s.id=m.species_id "
            if growth_form:
                extra += " AND s.growth_form=?"
                values.append(growth_form)
            if chinese_only:
                extra += " AND s.has_chinese_name=1"
            name_columns = ("t.chinese_name AS chineseName,t.name_source AS nameSource,"
                            if has_names else "NULL AS chineseName,NULL AS nameSource,")
            query = (
                "SELECT t.id,t.name," + name_columns +
                "t.rank,count(DISTINCT m.species_id) AS count FROM taxa t "
                f"JOIN membership m ON m.taxon_id=t.id {join}"
                f"WHERE t.parent=? {extra} GROUP BY t.id"
            )
            total = db.execute(f"SELECT count(*) FROM ({query})", values).fetchone()[0]
            rows = db.execute(
                query + " ORDER BY t.name LIMIT ? OFFSET ?", [*values, limit, offset],
            ).fetchall()
        items = [{**dict(row), "nameSource": json.loads(row["nameSource"]) if row["nameSource"] else None}
                 for row in rows]
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    def labels(self, indices):
        if not indices:
            return []
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM labels WHERE idx IN ({','.join('?' for _ in indices)})",
                [int(index) for index in indices],
            ).fetchall()
        by_index = {row["idx"]: dict(row) for row in rows}
        return [by_index[int(index)] for index in indices]

    def plant_indices(self):
        with self.connect() as db:
            return [row[0] for row in db.execute(
                "SELECT idx FROM labels WHERE species_id IS NOT NULL ORDER BY idx",
            )]
