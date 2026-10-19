import os
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def scan_roots(roots, store):
    roots = [Path(r).as_posix() for r in roots]  # 统一正斜杠，missing 前缀匹配才可靠
    seen = set()
    todo = []
    known = store.known_fingerprints()
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in SUPPORTED_EXTS:
                    continue
                p = (Path(dirpath) / fn).as_posix()
                seen.add(p)
                st = os.stat(p)
                fp = (st.st_mtime, st.st_size)
                if known.get(p) == fp:
                    continue
                store.upsert_photo(p, st.st_mtime, st.st_size)
                todo.append(p)
    known = set(store.known_fingerprints().keys())
    missing = [p for p in known if p not in seen and any(p.startswith(r) for r in roots)]
    if missing:
        store.mark_missing(missing)
    return todo
