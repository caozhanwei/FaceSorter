from app.ui.persons_model import persons_display

def test_display_names():
    rows = [{"id": 3, "name": None, "pinned": 0, "locked": 0, "cnt": 5},
            {"id": 1, "name": "张三", "pinned": 1, "locked": 1, "cnt": 9}]
    out = persons_display(rows)
    assert [d["label"] for d in out] == ["张三", "人物 3"]
    assert out[0]["cnt"] == 9 and out[0]["pinned"] == 1

def test_face_ids_codec():
    from app.ui.persons_model import encode_face_ids, decode_face_ids
    assert decode_face_ids(encode_face_ids([3, 14, 159])) == [3, 14, 159]
    assert decode_face_ids(b"") == []
