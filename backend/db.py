import json
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "projects.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            filename    TEXT NOT NULL DEFAULT '',
            job_id      TEXT NOT NULL DEFAULT '',
            segments    TEXT NOT NULL DEFAULT '[]',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)
    return conn


def save_project(project_id: str, name: str, filename: str,
                 job_id: str, segments: List[Dict[str, Any]]) -> None:
    now = time.time()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO projects (id, name, filename, job_id, segments, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, filename=excluded.filename,
                 job_id=excluded.job_id, segments=excluded.segments,
                 updated_at=excluded.updated_at""",
            (project_id, name, filename, job_id,
             json.dumps(segments, ensure_ascii=False), now, now),
        )


def list_projects() -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, filename, updated_at, segments FROM projects "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "filename": r["filename"],
        "updated_at": r["updated_at"],
        "segment_count": len(json.loads(r["segments"])),
    } for r in rows]


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        r = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "name": r["name"],
        "filename": r["filename"],
        "job_id": r["job_id"],
        "segments": json.loads(r["segments"]),
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def delete_project(project_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cur.rowcount > 0
