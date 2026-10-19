import os, time, pytest
from pathlib import Path
from app.store import Store
from app.scanner import scan_roots, SUPPORTED_EXTS

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")

def _touch(p: Path, size=10):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)

def test_new_changed_deleted(store, tmp_path):
    root = tmp_path / "lib"; _touch(root / "a.jpg"); _touch(root / "sub" / "b.png")
    _touch(root / "skip.txt")
    todo = scan_roots([str(root)], store)
    assert sorted(todo) == sorted([
        str(root / "a.jpg").replace("\\", "/"), str(root / "sub" / "b.png").replace("\\", "/")])
    # 全部入库后：无新增 → 空
    assert scan_roots([str(root)], store) == []
    # mtime 变化 → 重扫
    os.utime(root / "a.jpg", (time.time() + 10, time.time() + 10))
    assert scan_roots([str(root)], store) == [str(root / "a.jpg").replace("\\", "/")]
    # 删除 → missing
    (root / "a.jpg").unlink()
    scan_roots([str(root)], store)
    assert [r["path"] for r in store.photos_for_view("missing")] == [
        str(root / "a.jpg").replace("\\", "/")]
    # 重新出现 → 恢复 ok 并入 todo
    _touch(root / "a.jpg")
    assert scan_roots([str(root)], store) == [str(root / "a.jpg").replace("\\", "/")]
    assert store.photos_for_view("missing") == []

def test_multi_roots(store, tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    _touch(r1 / "a.jpg"); _touch(r2 / "b.jpg")
    todo = scan_roots([str(r1), str(r2)], store)
    assert len(todo) == 2
