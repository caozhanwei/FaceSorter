import sqlite3, numpy as np
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
  mtime REAL NOT NULL, size INTEGER NOT NULL,
  width INTEGER, height INTEGER, exif_taken_at TEXT,
  thumb_path TEXT, has_face INTEGER DEFAULT 0,
  status TEXT DEFAULT 'ok', scanned_at TEXT);
CREATE TABLE IF NOT EXISTS persons(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE,
  cover_face_id INTEGER, pinned INTEGER DEFAULT 0,
  locked INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY,
  photo_id INTEGER NOT NULL REFERENCES photos(id),
  x INTEGER, y INTEGER, w INTEGER, h INTEGER,
  quality REAL, feature BLOB,
  person_id INTEGER REFERENCES persons(id), source TEXT);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
"""

def pack_feature(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()

def unpack_feature(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

class StoreCorrupted(Exception):
    pass

FEATURE_VERSION = 3  # 1=LBP(1.0.x) 2=深度嵌入(1.1) 3=扫描期对齐裁剪(1.2)


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        try:
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.executescript(SCHEMA)
            if self.con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise StoreCorrupted(str(db_path))
            self._migrate_feature_version()
            self.con.commit()
        except sqlite3.DatabaseError as e:
            raise StoreCorrupted(str(db_path)) from e

    def _migrate_feature_version(self):
        v = self.con.execute("PRAGMA user_version").fetchone()[0]
        if v == FEATURE_VERSION:
            return
        # 旧特征（LBP）与新引擎嵌入不在同一空间，无法换算：清索引重扫。
        # 原照片文件永不被动；磁盘缩略图留作覆盖写，不算数据丢失。
        if v < FEATURE_VERSION:
            self.con.execute("DELETE FROM faces")
            self.con.execute("DELETE FROM persons")
            self.con.execute("DELETE FROM photos")
            self.con.execute(f"PRAGMA user_version={FEATURE_VERSION}")

    def _exec(self, sql, args=()):
        cur = self.con.execute(sql, args); self.con.commit(); return cur

    def upsert_photo(self, path, mtime, size):
        cur = self._exec(
            "UPDATE photos SET mtime=?, size=?, status='ok' WHERE path=?",
            (mtime, size, path))
        if cur.rowcount == 0:
            cur = self._exec(
                "INSERT INTO photos(path, mtime, size) VALUES(?,?,?)",
                (path, mtime, size))
            return cur.lastrowid
        # UPDATE 分支的 lastrowid 不可靠（返回的是连接上最近一次 INSERT 的 rowid）
        return self.con.execute(
            "SELECT id FROM photos WHERE path=?", (path,)).fetchone()[0]

    def update_photo_meta(self, photo_id, width=None, height=None,
                          exif_taken_at=None, thumb_path=None,
                          has_face=None, status=None):
        sets, args = [], []
        for col, val in [("width", width), ("height", height),
                         ("exif_taken_at", exif_taken_at),
                         ("thumb_path", thumb_path), ("has_face", has_face),
                         ("status", status)]:
            if val is not None:
                sets.append(f"{col}=?"); args.append(val)
        if sets:
            args.append(photo_id)
            self._exec(f"UPDATE photos SET {', '.join(sets)} WHERE id=?", args)

    def known_fingerprints(self):
        rows = self.con.execute(
            "SELECT path, mtime, size FROM photos WHERE status != 'missing'").fetchall()
        return {r["path"]: (r["mtime"], r["size"]) for r in rows}

    def mark_missing(self, paths):
        self.con.executemany(
            "UPDATE photos SET status='missing' WHERE path=?", [(p,) for p in paths])
        self.con.commit()

    def photos_for_view(self, view, name_filter=None):
        cond, args = {"all": "status='ok'", "with_faces": "status='ok' AND has_face=1",
                      "no_faces": "status='ok' AND has_face=0",
                      "bad": "status='bad'",
                      "missing": "status='missing'"}[view], []
        if name_filter:
            cond += " AND path LIKE ?"; args.append(f"%{name_filter}%")
        return self.con.execute(
            f"SELECT * FROM photos WHERE {cond} ORDER BY path", args).fetchall()

    def add_face(self, photo_id, x, y, w, h, quality, feature):
        cur = self._exec(
            "INSERT INTO faces(photo_id,x,y,w,h,quality,feature) VALUES(?,?,?,?,?,?,?)",
            (photo_id, x, y, w, h, quality, feature))
        return cur.lastrowid

    def set_face_person(self, face_id, person_id, source):
        self._exec("UPDATE faces SET person_id=?, source=? WHERE id=?",
                   (person_id, source, face_id))

    _FACE_COLS = ("f.id AS id, f.photo_id AS photo_id, x, y, w, h, quality,"
                  " person_id, source, p.path AS path, p.thumb_path AS thumb_path,"
                  " p.width AS pw, p.height AS ph, p.exif_taken_at AS exif_taken_at")

    def faces_of_person(self, person_id):
        return self.con.execute(
            f"SELECT {self._FACE_COLS} FROM faces f JOIN photos p ON p.id=f.photo_id"
            " WHERE f.person_id=? ORDER BY p.exif_taken_at, p.path", (person_id,)).fetchall()

    def photos_of_person(self, person_id):
        """人物的照片堆：合照出现在其中每个人的堆里（经 faces 多对多）。"""
        return self.con.execute(
            "SELECT p.* FROM photos p JOIN faces f ON f.photo_id=p.id"
            " WHERE f.person_id=? AND p.status='ok'"
            " GROUP BY p.id ORDER BY p.exif_taken_at, p.path", (person_id,)).fetchall()

    def face_crops_of_person(self, person_id, limit=16):
        """平均脸采样：质量（清晰度）降序的脸部裁剪 id。"""
        return self.con.execute(
            "SELECT id, quality FROM faces WHERE person_id=?"
            " ORDER BY quality DESC LIMIT ?", (person_id, limit)).fetchall()

    def faces_without_person(self):
        return self.con.execute(
            f"SELECT {self._FACE_COLS} FROM faces f JOIN photos p ON p.id=f.photo_id"
            " WHERE f.person_id IS NULL ORDER BY p.path").fetchall()

    def feature_by_face(self, face_id):
        b = self.con.execute(
            "SELECT feature FROM faces WHERE id=?", (face_id,)).fetchone()["feature"]
        return unpack_feature(b) if b is not None else None

    def photo_of_face(self, face_id):
        return self.con.execute(
            "SELECT p.* FROM photos p JOIN faces f ON f.photo_id=p.id WHERE f.id=?",
            (face_id,)).fetchone()

    def person_ids_of_photo(self, photo_id):
        rows = self.con.execute(
            "SELECT DISTINCT person_id FROM faces WHERE photo_id=? AND person_id IS NOT NULL",
            (photo_id,)).fetchall()
        return {r["person_id"] for r in rows}

    def create_person(self, name=None):
        cur = self._exec("INSERT INTO persons(name) VALUES(?)", (name,))
        return cur.lastrowid

    def rename_person(self, person_id, name):
        self._exec("UPDATE persons SET name=?, locked=1 WHERE id=?", (name, person_id))

    def merge_persons(self, src_id, dst_id):
        self._exec("UPDATE faces SET person_id=? WHERE person_id=?", (dst_id, src_id))
        self._exec("DELETE FROM persons WHERE id=?", (src_id,))

    def lock_person(self, person_id):
        self._exec("UPDATE persons SET locked=1 WHERE id=?", (person_id,))

    def set_pinned(self, person_id, pinned):
        self._exec("UPDATE persons SET pinned=? WHERE id=?", (1 if pinned else 0, person_id))

    def person_snapshot(self, person_id):
        rows = self.con.execute(
            "SELECT f.feature, p.exif_taken_at FROM faces f JOIN photos p ON p.id=f.photo_id"
            " WHERE f.person_id=? AND f.feature IS NOT NULL"
            " ORDER BY p.exif_taken_at", (person_id,)).fetchall()
        if not rows:
            return None, None
        mat = np.stack([unpack_feature(r["feature"]) for r in rows])
        return mat.mean(axis=0), rows[-1]["exif_taken_at"]

    def persons_with_counts(self):
        return self.con.execute(
            "SELECT p.id, p.name, p.pinned, p.locked, COUNT(f.id) AS cnt"
            " FROM persons p LEFT JOIN faces f ON f.person_id=p.id"
            " GROUP BY p.id ORDER BY p.pinned DESC, cnt DESC").fetchall()

    def count_photos(self, status="ok"):
        return self.con.execute(
            "SELECT COUNT(*) FROM photos WHERE status=?", (status,)).fetchone()[0]
