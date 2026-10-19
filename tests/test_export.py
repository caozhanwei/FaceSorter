import shutil
from pathlib import Path
from app.store import Store
from app.export import export_person

def test_export_copies_with_collision(tmp_path):
    store = Store(tmp_path / "t.db")
    src1 = tmp_path / "a.jpg"; src1.write_bytes(b"AAA"); src1m = src1
    src2 = tmp_path / "b.jpg"; src2.write_bytes(b"BBB")
    # 两照片各含一张同人的脸
    p1 = store.upsert_photo(str(src1).replace("\\", "/"), 1.0, 3)
    p2 = store.upsert_photo(str(src2).replace("\\", "/"), 1.0, 3)
    f1 = store.add_face(p1, 0, 0, 80, 80, 1.0, b"x")
    f2 = store.add_face(p2, 0, 0, 80, 80, 1.0, b"x")
    per = store.create_person("张三")
    store.set_face_person(f1, per, "manual"); store.set_face_person(f2, per, "manual")
    dest = tmp_path / "out"
    out = export_person(store, per, dest)
    assert len(out) == 2
    d = dest / "张三"
    assert (d / "a.jpg").read_bytes() == b"AAA"
    # 再导一次 → 重名 -2
    export_person(store, per, dest)
    assert (d / "a-2.jpg").exists()
    # 原文件未动
    assert src1.read_bytes() == b"AAA"
