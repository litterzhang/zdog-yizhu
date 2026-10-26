"""Bounded versioned disk cache of public species text vectors, never image features."""

import hashlib
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class TextVectorCache:
  def __init__(self, path: Path, namespace: str, *, dimension=768, limit=10000):
    if dimension < 1 or limit < 1:
      raise ValueError("Positive cache dimension and limit required.")
    self.path, self.namespace, self.dimension, self.limit = path, namespace, dimension, limit
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
      db.execute("CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY,value BLOB NOT NULL,used REAL NOT NULL)")
      db.execute("CREATE INDEX IF NOT EXISTS vectors_used ON vectors(used)")

  def key(self, name: str) -> str:
    return hashlib.sha256((self.namespace + "\0" + name).encode()).hexdigest()

  def get(self, name: str):
    import numpy as np

    key = self.key(name)
    with closing(sqlite3.connect(self.path)) as db, db:
      row = db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
      if row is None:
        return None
      raw = row[0]
      if len(raw) != self.dimension * 4:
        db.execute("DELETE FROM vectors WHERE key=?", (key,))
        return None
      vector = np.frombuffer(raw, dtype="<f4").copy()
      if not np.isfinite(vector).all() or not 0.999 <= float(np.linalg.norm(vector)) <= 1.001:
        db.execute("DELETE FROM vectors WHERE key=?", (key,))
        return None
      db.execute("UPDATE vectors SET used=? WHERE key=?", (time.time(), key))
    return vector

  def put(self, name: str, vector) -> None:
    import numpy as np

    value = np.asarray(vector, dtype="<f4")
    if (value.shape != (self.dimension,) or not np.isfinite(value).all()
        or not 0.999 <= float(np.linalg.norm(value)) <= 1.001):
      raise ValueError("Cache only finite normalized text vectors.")
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                 (self.key(name), value.tobytes(), time.time()))
      db.execute("""
        DELETE FROM vectors WHERE key IN (
          SELECT key FROM vectors ORDER BY used DESC,key LIMIT -1 OFFSET ?
        )
      """, (self.limit,))
