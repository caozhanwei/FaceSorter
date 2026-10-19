MIME_FACE_IDS = "application/x-facesorter-faceids"

def persons_display(rows):
    out = []
    for r in rows:
        out.append({"id": r["id"], "label": r["name"] or f"人物 {r['id']}",
                    "pinned": r["pinned"], "locked": r["locked"], "cnt": r["cnt"]})
    # 置顶优先、计数降序（与 store.persons_with_counts 的排序口径一致）
    out.sort(key=lambda d: (not d["pinned"], -d["cnt"]))
    return out

def encode_face_ids(ids) -> bytes:
    return ",".join(str(i) for i in ids).encode()

def decode_face_ids(data: bytes) -> list[int]:
    return [int(x) for x in data.decode().split(",") if x]
