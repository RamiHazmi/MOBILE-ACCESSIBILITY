"""
memory_service.py — SQLite-backed object location memory for WExist.

Tables
──────
  object_sightings(id, user_id, object_name, scene_description, gps_lat, gps_lng, saved_at)
  user_profile(id, user_id, name, created_at)
  named_locations(id, user_id, label, gps_lat, gps_lng, saved_at)

Public API
──────────
  save_sighting(name, lat, lng, scene_description, user_id)
  get_last_sighting(name, user_id)  → dict | None
  get_all_sightings(name, user_id)  → list[dict]
  get_user_name(user_id)            → str
  save_user_name(name, user_id)
  save_named_location(label, lat, lng, user_id)
  get_named_location(label, user_id) → dict | None
  delete_named_location(label, user_id) → bool
  relative_time(iso_ts)             → human-readable string
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Database path ─────────────────────────────────────────────
_HERE    = Path(__file__).parent
_DB_PATH = str(os.environ.get("MEMORY_DB_PATH",
                               _HERE / "memory.db"))

_lock = threading.Lock()   # one writer at a time


# ── Schema (fresh install) ────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS object_sightings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT    NOT NULL DEFAULT '',
    object_name       TEXT    NOT NULL,
    scene_description TEXT,
    gps_lat           REAL,
    gps_lng           REAL,
    saved_at          TEXT    NOT NULL   -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_object_user_name
    ON object_sightings (user_id, object_name);

CREATE TABLE IF NOT EXISTS user_profile (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT    NOT NULL DEFAULT '',
    name       TEXT    NOT NULL,
    created_at TEXT    NOT NULL   -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS named_locations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT    NOT NULL DEFAULT '',
    label      TEXT    NOT NULL,   -- e.g. "home", "work", "pharmacy"
    gps_lat    REAL    NOT NULL,
    gps_lng    REAL    NOT NULL,
    saved_at   TEXT    NOT NULL    -- ISO-8601 UTC
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_named_location_user_label
    ON named_locations (user_id, label);

CREATE TABLE IF NOT EXISTS personal_objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT '',
    name        TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    image_b64   TEXT,
    saved_at    TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_personal_obj_user_name
    ON personal_objects (user_id, name);

CREATE TABLE IF NOT EXISTS personal_persons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL DEFAULT '',
    name             TEXT    NOT NULL,
    relationship     TEXT    NOT NULL DEFAULT '',
    face_description TEXT    NOT NULL DEFAULT '',
    image_b64        TEXT,
    saved_at         TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_personal_person_user_name
    ON personal_persons (user_id, name);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    try:
        with _lock:
            conn = _connect()
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        print(f"[MemoryService] DB ready → {_DB_PATH}")
    except Exception as e:
        print(f"[MemoryService] DB init failed (non-fatal): {e}")


def _migrate_db() -> None:
    """Add user_id column to existing tables if missing, fix unique index."""
    try:
        with _lock:
            conn = _connect()
            for table in ("object_sightings", "user_profile", "named_locations"):
                cols = [row[1] for row in
                        conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if "user_id" not in cols:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
                    )
                    print(f"[MemoryService] migrated: added user_id to {table}")
            # Replace single-label unique index with per-user composite index
            conn.execute("DROP INDEX IF EXISTS idx_named_location_label")
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_named_location_user_label
                    ON named_locations (user_id, label)
            """)
            # personal_objects table (new)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personal_objects (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL DEFAULT '',
                    name        TEXT    NOT NULL,
                    description TEXT    NOT NULL DEFAULT '',
                    image_b64   TEXT,
                    saved_at    TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_personal_obj_user_name
                    ON personal_objects (user_id, name)
            """)
            # personal_persons table (new)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personal_persons (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id          TEXT    NOT NULL DEFAULT '',
                    name             TEXT    NOT NULL,
                    relationship     TEXT    NOT NULL DEFAULT '',
                    face_description TEXT    NOT NULL DEFAULT '',
                    image_b64        TEXT,
                    saved_at         TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_personal_person_user_name
                    ON personal_persons (user_id, name)
            """)
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[MemoryService] migration error (non-fatal): {e}")


# Initialise on import
_init_db()
_migrate_db()


# ── User profile ──────────────────────────────────────────────

def get_user_name(user_id: str = "") -> str:
    """Return the stored user name for this device, or empty string."""
    try:
        with _lock:
            conn = _connect()
            row  = conn.execute(
                "SELECT name FROM user_profile WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            conn.close()
        return row["name"] if row else ""
    except Exception as e:
        print(f"[MemoryService] get_user_name error: {e}")
        return ""


def save_user_name(name: str, user_id: str = "") -> None:
    """Persist the user's name for this device. Replaces any previous entry."""
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            conn = _connect()
            conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
            conn.execute(
                "INSERT INTO user_profile (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name.strip(), ts),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] user name saved: '{name}' for user={user_id[:8]}…")
    except Exception as e:
        print(f"[MemoryService] save_user_name error: {e}")


# ── Named locations ───────────────────────────────────────────

def save_named_location(label: str, lat: float, lng: float,
                        user_id: str = "") -> None:
    """Save or update a named GPS location for this device."""
    try:
        ts  = datetime.now(timezone.utc).isoformat()
        lbl = label.strip().lower()
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO named_locations (user_id, label, gps_lat, gps_lng, saved_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, label) DO UPDATE SET
                       gps_lat=excluded.gps_lat,
                       gps_lng=excluded.gps_lng,
                       saved_at=excluded.saved_at""",
                (user_id, lbl, lat, lng, ts),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] saved location '{label}' @ {lat},{lng} user={user_id[:8]}…")
    except Exception as e:
        print(f"[MemoryService] save_named_location error: {e}")


def get_named_location(label: str, user_id: str = "") -> Optional[dict]:
    """Return a named location record for this device, or None."""
    try:
        lbl = label.strip().lower()
        with _lock:
            conn = _connect()
            row  = conn.execute(
                "SELECT * FROM named_locations WHERE user_id = ? AND label = ?",
                (user_id, lbl),
            ).fetchone()
            conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[MemoryService] get_named_location error: {e}")
        return None


def delete_named_location(label: str, user_id: str = "") -> bool:
    """Delete a named location for this device. Returns True if deleted."""
    try:
        lbl = label.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "DELETE FROM named_locations WHERE user_id = ? AND label = ?",
                (user_id, lbl),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            conn.close()
        print(f"[MemoryService] deleted location '{label}': {deleted}")
        return deleted
    except Exception as e:
        print(f"[MemoryService] delete_named_location error: {e}")
        return False


def get_all_named_locations(user_id: str = "") -> list[dict]:
    """Return all named locations for this device."""
    try:
        with _lock:
            conn = _connect()
            rows = conn.execute(
                "SELECT label, gps_lat, gps_lng FROM named_locations WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[MemoryService] get_all_named_locations error: {e}")
        return []


# ── Write ─────────────────────────────────────────────────────

def save_sighting(
    name:              str,
    lat:               Optional[float] = None,
    lng:               Optional[float] = None,
    scene_description: str             = "",
    user_id:           str             = "",
) -> None:
    """
    Append one sighting record.  Called silently from find_object node.
    Never raises — errors are logged and swallowed so the main flow
    is never interrupted.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO object_sightings
                   (user_id, object_name, scene_description, gps_lat, gps_lng, saved_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, name.strip().lower(), scene_description, lat, lng, ts),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] saved sighting: '{name}' @ {lat},{lng}")
    except Exception as e:
        print(f"[MemoryService] save_sighting error (non-fatal): {e}")


# ── Read ──────────────────────────────────────────────────────

def get_last_sighting(name: str, user_id: str = "") -> Optional[dict]:
    """Return the most recent sighting for *name* by this device, or None."""
    try:
        with _lock:
            conn = _connect()
            row  = conn.execute(
                """SELECT * FROM object_sightings
                   WHERE user_id = ? AND object_name = ?
                   ORDER BY saved_at DESC LIMIT 1""",
                (user_id, name.strip().lower()),
            ).fetchone()
            conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[MemoryService] get_last_sighting error: {e}")
        return None


def get_all_sightings(name: str, user_id: str = "") -> list[dict]:
    """Return all sightings for *name* by this device, newest first."""
    try:
        with _lock:
            conn = _connect()
            rows = conn.execute(
                """SELECT * FROM object_sightings
                   WHERE user_id = ? AND object_name = ?
                   ORDER BY saved_at DESC""",
                (user_id, name.strip().lower()),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[MemoryService] get_all_sightings error: {e}")
        return []


# ── Personal objects ──────────────────────────────────────────

def save_personal_object(
    name:        str,
    description: str = "",
    image_b64:   str = "",
    user_id:     str = "",
) -> None:
    """Save or update a personal object (keys, bag, person…) with its description."""
    try:
        ts = datetime.now(timezone.utc).isoformat()
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO personal_objects
                       (user_id, name, description, image_b64, saved_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, name) DO UPDATE SET
                       description=excluded.description,
                       image_b64=excluded.image_b64,
                       saved_at=excluded.saved_at""",
                (user_id, nm, description.strip(), image_b64 or "", ts),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] saved personal object '{name}' for user={user_id[:8]}…")
    except Exception as e:
        print(f"[MemoryService] save_personal_object error: {e}")


def get_personal_object(name: str, user_id: str = "") -> Optional[dict]:
    """Return a saved personal object by name, or None."""
    try:
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            row  = conn.execute(
                "SELECT * FROM personal_objects WHERE user_id = ? AND name = ?",
                (user_id, nm),
            ).fetchone()
            conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[MemoryService] get_personal_object error: {e}")
        return None


def get_all_personal_objects(user_id: str = "") -> list[dict]:
    """Return all saved personal objects for this user."""
    try:
        with _lock:
            conn = _connect()
            rows = conn.execute(
                "SELECT name, description FROM personal_objects WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[MemoryService] get_all_personal_objects error: {e}")
        return []


def get_all_sightings_for_user(user_id: str = "", limit: int = 40) -> list[dict]:
    """Return the most recent sightings across all objects for this user."""
    try:
        with _lock:
            conn = _connect()
            rows = conn.execute(
                """SELECT * FROM object_sightings
                   WHERE user_id = ?
                   ORDER BY saved_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[MemoryService] get_all_sightings_for_user error: {e}")
        return []


def delete_sighting(sighting_id: int, user_id: str = "") -> bool:
    """Delete a specific sighting by id (scoped to user). Returns True if deleted."""
    try:
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "DELETE FROM object_sightings WHERE id = ? AND user_id = ?",
                (sighting_id, user_id),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            conn.close()
        print(f"[MemoryService] deleted sighting id={sighting_id}: {deleted}")
        return deleted
    except Exception as e:
        print(f"[MemoryService] delete_sighting error: {e}")
        return False


def delete_personal_object(name: str, user_id: str = "") -> bool:
    """Delete a saved personal object. Returns True if deleted."""
    try:
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "DELETE FROM personal_objects WHERE user_id = ? AND name = ?",
                (user_id, nm),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            conn.close()
        print(f"[MemoryService] deleted personal object '{name}': {deleted}")
        return deleted
    except Exception as e:
        print(f"[MemoryService] delete_personal_object error: {e}")
        return False


# ── Personal persons ─────────────────────────────────────────

def save_personal_person(
    name:             str,
    relationship:     str = "person",
    image_b64:        str = "",
    face_description: str = "",
    user_id:          str = "",
) -> None:
    """Save or update a personal person (friend, family…) with face description."""
    try:
        ts = datetime.now(timezone.utc).isoformat()
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO personal_persons
                       (user_id, name, relationship, face_description, image_b64, saved_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, name) DO UPDATE SET
                       relationship=excluded.relationship,
                       face_description=excluded.face_description,
                       image_b64=excluded.image_b64,
                       saved_at=excluded.saved_at""",
                (user_id, nm, relationship.strip(), face_description.strip(), image_b64 or "", ts),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] saved person '{name}' ({relationship}) for user={user_id[:8]}…")
    except Exception as e:
        print(f"[MemoryService] save_personal_person error: {e}")


def get_personal_person(name: str, user_id: str = "") -> Optional[dict]:
    """Return a saved person by name, or None."""
    try:
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            row  = conn.execute(
                "SELECT * FROM personal_persons WHERE user_id = ? AND name = ?",
                (user_id, nm),
            ).fetchone()
            conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[MemoryService] get_personal_person error: {e}")
        return None


def get_all_personal_persons(user_id: str = "") -> list[dict]:
    """Return all saved persons for this user, including image_b64 for face comparison."""
    try:
        with _lock:
            conn = _connect()
            rows = conn.execute(
                "SELECT name, relationship, face_description, image_b64 "
                "FROM personal_persons WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[MemoryService] get_all_personal_persons error: {e}")
        return []


def delete_personal_person(name: str, user_id: str = "") -> bool:
    """Delete a saved person. Returns True if deleted."""
    try:
        nm = name.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "DELETE FROM personal_persons WHERE user_id = ? AND name = ?",
                (user_id, nm),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            conn.close()
        print(f"[MemoryService] deleted person '{name}': {deleted}")
        return deleted
    except Exception as e:
        print(f"[MemoryService] delete_personal_person error: {e}")
        return False


# ── Rename helpers ────────────────────────────────────────────

def rename_personal_person(old_name: str, new_name: str, user_id: str = "") -> bool:
    """Rename a saved person. Returns True if the row was found and updated."""
    try:
        old = old_name.strip().lower()
        new = new_name.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "UPDATE personal_persons SET name=? WHERE user_id=? AND name=?",
                (new, user_id, old),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] renamed person '{old}' → '{new}': {cur.rowcount > 0}")
        return cur.rowcount > 0
    except Exception as e:
        print(f"[MemoryService] rename_personal_person error: {e}")
        return False


def rename_personal_object(old_name: str, new_name: str, user_id: str = "") -> bool:
    """Rename a saved personal object. Returns True if found and updated."""
    try:
        old = old_name.strip().lower()
        new = new_name.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "UPDATE personal_objects SET name=? WHERE user_id=? AND name=?",
                (new, user_id, old),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] renamed object '{old}' → '{new}': {cur.rowcount > 0}")
        return cur.rowcount > 0
    except Exception as e:
        print(f"[MemoryService] rename_personal_object error: {e}")
        return False


def rename_named_location(old_label: str, new_label: str, user_id: str = "") -> bool:
    """Rename a saved location. Returns True if found and updated."""
    try:
        old = old_label.strip().lower()
        new = new_label.strip().lower()
        with _lock:
            conn = _connect()
            cur  = conn.execute(
                "UPDATE named_locations SET label=? WHERE user_id=? AND label=?",
                (new, user_id, old),
            )
            conn.commit()
            conn.close()
        print(f"[MemoryService] renamed location '{old}' → '{new}': {cur.rowcount > 0}")
        return cur.rowcount > 0
    except Exception as e:
        print(f"[MemoryService] rename_named_location error: {e}")
        return False


# ── Reverse geocode (reuses the same helper as orchestrator) ──

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """Return a short human-readable location string, or None."""
    try:
        import requests as _req
        r = _req.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json"},
            headers={"User-Agent": "WExistApp/1.0"},
            timeout=6,
        )
        addr  = r.json().get("address", {})
        parts = []
        road  = (addr.get("road") or addr.get("pedestrian")
                 or addr.get("footway"))
        if road:
            parts.append(road)
        area  = (addr.get("suburb") or addr.get("neighbourhood")
                 or addr.get("city_district") or addr.get("town")
                 or addr.get("city") or addr.get("county"))
        if area:
            parts.append(area)
        return ", ".join(parts) if parts else r.json().get("display_name", "")[:60]
    except Exception:
        return None


# ── Relative time ─────────────────────────────────────────────

def relative_time(iso_ts: str, lang: str = "en") -> str:
    """
    Convert an ISO-8601 UTC timestamp to a human-readable relative
    string in the requested language.

    Examples (en):  "3 minutes ago", "2 hours ago", "yesterday", "3 days ago"
    """
    try:
        then  = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        now   = datetime.now(timezone.utc)
        delta = now - then
        secs  = int(delta.total_seconds())

        if secs < 60:
            n = max(secs, 1)
            return _rt("seconds_ago", lang, n=n)
        if secs < 3600:
            n = secs // 60
            return _rt("minutes_ago", lang, n=n)
        if secs < 86400:
            n = secs // 3600
            return _rt("hours_ago", lang, n=n)
        if secs < 172800:
            return _rt("yesterday", lang)
        n = secs // 86400
        return _rt("days_ago", lang, n=n)
    except Exception:
        return iso_ts   # fallback: raw timestamp


# ── Relative-time phrase table ────────────────────────────────

_RT: dict[str, dict[str, str]] = {
    "seconds_ago": {
        "en": "a few seconds ago",
        "fr": "il y a quelques secondes",
        "ar": "منذ ثوانٍ قليلة",
        "tn": "من وقت قريب",
    },
    "minutes_ago": {
        "en": "{n} minute ago" if False else "{n} minutes ago",   # handled below
        "fr": "il y a {n} minute",
        "ar": "منذ {n} دقيقة",
        "tn": "من {n} دقيقة",
    },
    "hours_ago": {
        "en": "{n} hours ago",
        "fr": "il y a {n} heure",
        "ar": "منذ {n} ساعة",
        "tn": "من {n} ساعة",
    },
    "yesterday": {
        "en": "yesterday",
        "fr": "hier",
        "ar": "أمس",
        "tn": "البارح",
    },
    "days_ago": {
        "en": "{n} days ago",
        "fr": "il y a {n} jours",
        "ar": "منذ {n} أيام",
        "tn": "من {n} أيام",
    },
}


def _rt(key: str, lang: str, **kwargs) -> str:
    bundle   = _RT.get(key, {})
    template = bundle.get(lang) or bundle.get("en") or key
    # Special-case English singular
    if key == "minutes_ago" and lang == "en":
        n = kwargs.get("n", 0)
        template = "{n} minute ago" if n == 1 else "{n} minutes ago"
    if key == "hours_ago" and lang == "en":
        n = kwargs.get("n", 0)
        template = "{n} hour ago" if n == 1 else "{n} hours ago"
    try:
        return template.format(**kwargs)
    except Exception:
        return template