from app.features import hellinger, parse_time

class Clusterer:
    def __init__(self, store, threshold=0.5, time_window_min=30, time_boost=1.1):
        self.store = store
        self.threshold = threshold
        self.time_window_min = time_window_min
        self.time_boost = time_boost
        self._cache = {}  # person_id -> (prototype, last_exif_str)

    def invalidate(self, person_id):
        self._cache.pop(person_id, None)

    def _snapshot(self, person_id):
        if person_id not in self._cache:
            self._cache[person_id] = self.store.person_snapshot(person_id)
        return self._cache[person_id]

    def assign(self, face_id):
        feat = self.store.feature_by_face(face_id)
        if feat is None:
            raise ValueError(f"face {face_id} has no feature")
        photo = self.store.photo_of_face(face_id)
        taken = parse_time(photo["exif_taken_at"])
        same = self.store.person_ids_of_photo(photo["id"])
        best_pid, best_d = None, self.threshold
        for row in self.store.persons_with_counts():
            pid = row["id"]
            if pid in same or row["cnt"] == 0:
                continue
            proto, last_s = self._snapshot(pid)
            if proto is None:
                continue
            d = hellinger(feat, proto)
            last = parse_time(last_s)
            if (taken and last and self.time_boost > 1.0
                    and abs((taken - last).total_seconds())
                    <= self.time_window_min * 60):
                d = d / self.time_boost
            if d < best_d:
                best_pid, best_d = pid, d
        if best_pid is None:
            best_pid = self.store.create_person()
            self.invalidate(best_pid)
        self.store.set_face_person(face_id, best_pid, "auto")
        self.invalidate(best_pid)
        return best_pid

    def recluster_auto(self):
        rows = self.store.con.execute(
            "SELECT id FROM faces WHERE source='auto'").fetchall()
        self.store.con.execute("UPDATE faces SET person_id=NULL WHERE source='auto'")
        self.store.con.commit()
        self._cache.clear()
        for r in rows:
            self.assign(r["id"])
        # 清掉重排后没有脸且未锁定的空堆
        self.store.con.execute(
            "DELETE FROM persons WHERE locked=0 AND id NOT IN"
            " (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
        self.store.con.commit()
