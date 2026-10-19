import shutil
from pathlib import Path


def export_person(store, person_id, dest_dir, progress=None):
    """把某人的全部照片原文件复制到 dest/人名/ 下（重名 -2 递增后缀）。

    只复制副本，绝不移动/删除原文件；按 photo_id 去重；源不存在的跳过。
    返回成功复制的目标路径列表。
    """
    label = None
    for r in store.persons_with_counts():
        if r["id"] == person_id:
            label = r["name"] or f"人物 {r['id']}"
            break
    if label is None:
        return []
    target = Path(dest_dir) / label
    target.mkdir(parents=True, exist_ok=True)
    done, copied = set(), []
    rows = store.faces_of_person(person_id)
    for i, r in enumerate(rows):
        if r["photo_id"] in done:
            continue
        done.add(r["photo_id"])
        src = Path(r["path"])
        if not src.exists():
            continue
        dst = target / src.name
        n = 2
        stem, suf = dst.stem, dst.suffix
        while dst.exists():
            dst = target / f"{stem}-{n}{suf}"; n += 1
        shutil.copy2(src, dst)
        copied.append(str(dst))
        if progress:
            progress(i + 1, len(rows))
    return copied
