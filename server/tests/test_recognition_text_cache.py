import sqlite3
from contextlib import closing

import pytest

from yizhu_server.recognition_text_cache import TextVectorCache

np = pytest.importorskip("numpy")


def test_disk_vectors_persist_and_are_namespaced_by_model_configuration(tmp_path):
  path = tmp_path / "vectors.sqlite3"
  cache = TextVectorCache(path, "model-template-v1", dimension=3)
  cache.put("Example plant", np.array([1, 0, 0]))
  restored = TextVectorCache(path, "model-template-v1", dimension=3)
  assert np.array_equal(restored.get("Example plant"), [1, 0, 0])
  assert TextVectorCache(path, "model-template-v2", dimension=3).get("Example plant") is None
  assert cache.get("Unknown plant") is None


def test_disk_cache_evicts_old_entries_and_never_stores_private_paths(tmp_path, monkeypatch):
  from yizhu_server import recognition_text_cache

  ticks = iter(range(100))
  monkeypatch.setattr(recognition_text_cache.time, "time", lambda: next(ticks))
  path = tmp_path / "vectors.sqlite3"
  cache = TextVectorCache(path, "v1", dimension=2, limit=2)
  cache.put("first", [1, 0])
  cache.put("second", [0, 1])
  cache.get("first")
  cache.put("third", [1, 0])
  assert cache.get("second") is None
  assert cache.get("first") is not None
  with closing(sqlite3.connect(path)) as db:
    rows = db.execute("SELECT key,length(value) FROM vectors").fetchall()
  assert len(rows) == 2
  assert all(len(key) == 64 and length == 8 for key, length in rows)


@pytest.mark.parametrize("value", [[0, 0], [2, 0], [float("nan"), 0], [1, 0, 0]])
def test_bad_vectors_cannot_enter_cache(tmp_path, value):
  cache = TextVectorCache(tmp_path / "cache", "v1", dimension=2)
  with pytest.raises(ValueError):
    cache.put("Example plant", value)


@pytest.mark.parametrize("raw", [b"short", np.array([np.nan, 0], dtype="<f4").tobytes()])
def test_corrupt_cached_vectors_are_removed_and_recomputed(tmp_path, raw):
  path = tmp_path / "cache"
  cache = TextVectorCache(path, "v1", dimension=2)
  with sqlite3.connect(path) as db:
    db.execute("INSERT INTO vectors VALUES (?,?,?)", (cache.key("plant"), raw, 1))
  assert cache.get("plant") is None
  cache.put("plant", [1, 0])
  assert np.array_equal(cache.get("plant"), [1, 0])
