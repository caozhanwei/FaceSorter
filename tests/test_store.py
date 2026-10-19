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
