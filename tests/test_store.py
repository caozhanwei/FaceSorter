import numpy as np, pytest
from pathlib import Path
from app.store import Store, pack_feature, unpack_feature

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")

def test_upsert_photo_idempotent(store):
    a = store.upsert_photo("x.jpg", 1.0, 10)
    b = store.upsert_photo("x.jpg", 2.0, 11)
    assert a == b
    fp = store.known_fingerprints()
    assert fp["x.jpg"] == (2.0, 11)

def test_face_person_flow(store):
    pid = store.upsert_photo("x.jpg", 1.0, 10)
    f = store.add_face(pid, 1, 2, 30, 40, 77.0, pack_feature(np.ones(8, np.float32)))
    store.set_face_person(f, None, "auto")
    assert [r["id"] for r in store.faces_without_person()] == [f]
    p = store.create_person()
    store.set_face_person(f, p, "manual")
    assert len(store.faces_of_person(p)) == 1
    assert store.person_ids_of_photo(pid) == {p}
    assert store.feature_by_face(f).shape == (8,)

def test_rename_locks_and_merge(store):
    pid = store.upsert_photo("x.jpg", 1.0, 10)
    f1 = store.add_face(pid, 0, 0, 10, 10, 1.0, pack_feature(np.zeros(4, np.float32)))
    f2 = store.add_face(pid, 10, 0, 10, 10, 1.0, pack_feature(np.ones(4, np.float32)))
    p1, p2 = store.create_person(), store.create_person()
    store.set_face_person(f1, p1, "auto"); store.set_face_person(f2, p2, "auto")
    store.rename_person(p1, "张三")
    assert store.persons_with_counts()[0]["locked"] == 1
    store.merge_persons(p2, p1)
    assert len(store.faces_of_person(p1)) == 2
    assert {r["id"] for r in store.persons_with_counts() if r["id"] == p2} == set()

def test_snapshot_prototype(store):
    pid = store.upsert_photo("x.jpg", 1.0, 10)
    store.update_photo_meta(pid, exif_taken_at="2026:01:02 10:00:00")
    f1 = store.add_face(pid, 0, 0, 10, 10, 1.0, pack_feature(np.zeros(4, np.float32)))
    f2 = store.add_face(pid, 0, 0, 10, 10, 1.0, pack_feature(np.ones(4, np.float32) * 2))
    p = store.create_person()
    store.set_face_person(f1, p, "auto"); store.set_face_person(f2, p, "auto")
    proto, last = store.person_snapshot(p)
    assert np.allclose(proto, np.ones(4, np.float32))
    assert last == "2026:01:02 10:00:00"

def test_views(store):
    pid = store.upsert_photo("a.jpg", 1.0, 10)
    store.update_photo_meta(pid, has_face=1)
    store.upsert_photo("b.jpg", 1.0, 10)
    bad = store.upsert_photo("c.jpg", 1.0, 10)
    store.update_photo_meta(bad, status="bad")
    assert [r["path"] for r in store.photos_for_view("all")] == ["a.jpg", "b.jpg"]
    assert [r["path"] for r in store.photos_for_view("with_faces")] == ["a.jpg"]
    assert [r["path"] for r in store.photos_for_view("no_faces")] == ["b.jpg"]
    assert [r["path"] for r in store.photos_for_view("bad")] == ["c.jpg"]

def test_corrupt_db_detected(tmp_path):
    from app.store import StoreCorrupted
    p = tmp_path / "bad.db"; p.write_bytes(b"not a sqlite file")
    with pytest.raises(StoreCorrupted):
        Store(p)


# ---- 特征格式版本化：1.0.x（LBP）→ 1.1（嵌入）→ 1.2（扫描期裁剪）逐级重扫 ----

def test_feature_format_migration_wipes_index(tmp_path):
    db = tmp_path / "t.db"
    s1 = Store(db)
    pid = s1.upsert_photo("a.jpg", 1.0, 1)
    s1.add_face(pid, 0, 0, 80, 80, 100.0, b"\x00" * 512)
    s1.create_person("张三")
    s1.con.execute("PRAGMA user_version=0")  # 模拟 1.0.x（LBP）库
    s1.con.commit(); s1.con.close()
    s2 = Store(db)
    assert s2.con.execute("PRAGMA user_version").fetchone()[0] == 3
    # 旧特征与新引擎不同空间：清脸/人物/照片行，重扫重建（原照片文件不动）
    assert s2.count_photos() == 0
    assert s2.con.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0
    assert s2.con.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0

def test_v2_db_wiped_for_face_crops(tmp_path):
    # 1.1（v2）的脸没有扫描期裁剪：升 1.2 须清索引重扫一次生成清晰头像
    db = tmp_path / "t.db"
    s1 = Store(db)
    pid = s1.upsert_photo("a.jpg", 1.0, 1)
    s1.add_face(pid, 0, 0, 80, 80, 100.0, b"\x00" * 128)
    s1.con.execute("PRAGMA user_version=2")  # 模拟 1.1.x 库
    s1.con.commit(); s1.con.close()
    s2 = Store(db)
    assert s2.con.execute("PRAGMA user_version").fetchone()[0] == 3
    assert s2.count_photos() == 0
    assert s2.con.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0

def test_fresh_db_stamps_version_and_keeps_data(tmp_path):
    db = tmp_path / "t.db"
    s1 = Store(db)
    assert s1.con.execute("PRAGMA user_version").fetchone()[0] == 3
    s1.upsert_photo("a.jpg", 1.0, 1)
    s1.con.commit(); s1.con.close()
    s2 = Store(db)
    assert s2.count_photos() == 1  # 同版本重开不清数据


# ---- 人物↔照片多对多（合照共享）与平均脸采样 ----

def test_photos_of_person_shared_group_photo(tmp_path):
    # 一张合照两个人都在：两个人的照片堆都含这张照片
    store = Store(tmp_path / "t.db")
    ph = store.upsert_photo("g:/group.jpg", 1.0, 1)
    f1 = store.add_face(ph, 0, 0, 60, 60, 100.0, None)
    f2 = store.add_face(ph, 100, 0, 60, 60, 100.0, None)
    pa = store.create_person("甲")
    pb = store.create_person("乙")
    store.set_face_person(f1, pa, "auto")
    store.set_face_person(f2, pb, "auto")
    assert [r["id"] for r in store.photos_of_person(pa)] == [ph]
    assert [r["id"] for r in store.photos_of_person(pb)] == [ph]

def test_face_crops_of_person_quality_order(tmp_path):
    # 平均脸采样：质量（清晰度）降序给前 N 张裁剪
    store = Store(tmp_path / "t.db")
    ph = store.upsert_photo("a.jpg", 1.0, 1)
    f_low = store.add_face(ph, 0, 0, 10, 10, 10.0, None)
    f_hi = store.add_face(ph, 0, 0, 20, 20, 900.0, None)
    f_mid = store.add_face(ph, 0, 0, 30, 30, 100.0, None)
    p = store.create_person("X")
    for f in (f_low, f_hi, f_mid):
        store.set_face_person(f, p, "auto")
    ids = [r["id"] for r in store.face_crops_of_person(p)]
    assert ids == [f_hi, f_mid, f_low]
