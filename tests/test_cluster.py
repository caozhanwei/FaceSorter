import numpy as np, pytest
from app.store import Store, pack_feature
from app.cluster import Clusterer

def _mk(store, feat, photo_path, exif=None):
    pid = store.upsert_photo(photo_path, 1.0, 1)
    store.update_photo_meta(pid, exif_taken_at=exif)
    return store.add_face(pid, 0, 0, 80, 80, 100.0, pack_feature(feat))

def _norm(v):
    v = np.asarray(v, np.float32); return v / np.linalg.norm(v)

@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "t.db")
    return store, Clusterer(store, threshold=0.5)

def test_similar_faces_same_person(env):
    store, cl = env
    a = _norm([1, 0, 0, 0]); b = _norm([0.95, 0.05, 0, 0]); c = _norm([0, 0, 1, 0])
    f1 = _mk(store, a, "1.jpg"); f2 = _mk(store, b, "2.jpg"); f3 = _mk(store, c, "3.jpg")
    assert cl.assign(f1) == cl.assign(f2)
    assert cl.assign(f3) != cl.assign(f1)

def test_same_photo_never_same_person(env):
    store, cl = env
    a = _norm([1, 0, 0, 0])
    f1 = _mk(store, a, "g.jpg"); f2 = _mk(store, a, "g.jpg")  # 同照片同特征(合影极端)
    p1, p2 = cl.assign(f1), cl.assign(f2)
    assert p1 != p2

def test_time_boost_merges_borderline(env):
    store, cl = env
    # 标准 hellinger(√向量再 L2 归一化)：d(a,b)≈0.520>0.5（无加成不并），÷1.1≈0.473<0.5（加成后并）
    a = _norm([1, 0, 0, 0]); b = _norm([0.7514, 0.6601, 0, 0])
    c = _norm([0, 1, 0, 0])
    f1 = _mk(store, a, "1.jpg", "2026:01:01 10:00:00")
    f2 = _mk(store, b, "2.jpg", "2026:01:01 10:10:00")     # 10 分钟内 → 加成后归并
    f3 = _mk(store, c, "3.jpg", "2026:02:01 10:00:00")     # 异距特征 → 不归并
    assert cl.assign(f1) == cl.assign(f2)
    assert cl.assign(f3) != cl.assign(f1)

def test_far_time_borderline_stays_split(env):
    store, cl = env
    a = _norm([1, 0, 0, 0]); b = _norm([0.7514, 0.6601, 0, 0])
    f1 = _mk(store, a, "1.jpg", "2026:01:01 10:00:00")
    f2 = _mk(store, b, "2.jpg", "2026:09:01 10:00:00")     # 同距特征、相隔 8 月 → 无加成不并
    assert cl.assign(f1) != cl.assign(f2)

def test_recluster_keeps_manual(env):
    store, cl = env
    a = _norm([1, 0, 0, 0])
    f1 = _mk(store, a, "1.jpg"); f2 = _mk(store, a, "2.jpg")
    p = cl.assign(f1); cl.assign(f2)
    other = store.create_person("李四")
    store.set_face_person(f2, other, "manual")
    cl.invalidate(p); cl.invalidate(other)
    cl.recluster_auto()
    rows = {r["id"]: r["source"] for r in store.faces_of_person(other)}
    assert rows[f2] == "manual"   # manual 归属永不被自动重排覆盖
    assert rows[f1] == "auto"     # auto 脸可经原型并入人工堆（spec §5.3）
