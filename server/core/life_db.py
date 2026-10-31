"""SQLite Life Command Matrix Engine (Tasks, Calendar Schedule, Habit Streaks)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.config import settings


class LifeCommandDB:
    """Manages tasks, calendar timeline blocks, and habit streaks in SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = db_path or settings.SQLITE_DB_PATH
        self._init_db()
        self._seed_defaults()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS life_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    priority TEXT DEFAULT 'normal',
                    status TEXT DEFAULT 'pending',
                    due_time TEXT DEFAULT 'Today',
                    linked_entity TEXT DEFAULT '[[Daily Priorities]]',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS life_schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_slot TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT DEFAULT 'Executive',
                    active INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS life_habits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    streak INTEGER DEFAULT 1,
                    completed_today INTEGER DEFAULT 0,
                    target TEXT DEFAULT 'Daily'
                );
                """
            )

    def _seed_defaults(self) -> None:
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM life_tasks").fetchone()[0]
            if count == 0:
                now = datetime.now().isoformat(timespec="seconds")
                default_tasks = [
                    ("Calibrate Silero VAD <180ms Barge-In Threshold", "critical", "pending", "14:00", "[[Full-Duplex Voice Engine]]", now),
                    ("Audit Google AI Studio Free Tier Token-Bucket Router", "high", "pending", "16:30", "[[Hybrid Cognitive Router]]", now),
                    ("Expand 2-Hop Obsidian Knowledge Graph Triples", "normal", "completed", "11:00", "[[Cortex Memory Core]]", now),
                    ("Verify Tailscale PWA Mobile Mesh Tunnel", "normal", "pending", "19:00", "[[Spatial HUD]]", now),
                ]
                conn.executemany(
                    "INSERT INTO life_tasks (title, priority, status, due_time, linked_entity, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    default_tasks,
                )

            sched_count = conn.execute("SELECT COUNT(*) FROM life_schedule").fetchone()[0]
            if sched_count == 0:
                default_schedule = [
                    ("09:00", "Morning Executive Briefing & Telemetry Sync", "Briefing", 0),
                    ("13:30", "Project J.A.R.V.I.S. Spatial HUD Architecture Build", "Deep Work", 1),
                    ("16:00", "Cortex Memory Core Graph Indexing Review", "Research", 0),
                    ("20:00", "Evening Systems Audit & Next-Day Priority Lock", "Operations", 0),
                ]
                conn.executemany(
                    "INSERT INTO life_schedule (time_slot, title, category, active) VALUES (?, ?, ?, ?)",
                    default_schedule,
                )

            habit_count = conn.execute("SELECT COUNT(*) FROM life_habits").fetchone()[0]
            if habit_count == 0:
                default_habits = [
                    ("Deep Systems Architecture Block (2h)", 14, 1, "Daily"),
                    ("Zero-Inbox & Knowledge Vault Pruning", 9, 1, "Daily"),
                    ("Physical Conditioning & Telemetry Check", 21, 0, "Daily"),
                ]
                conn.executemany(
                    "INSERT INTO life_habits (name, streak, completed_today, target) VALUES (?, ?, ?, ?)",
                    default_habits,
                )

    def add_task(
        self,
        title: str,
        priority: str = "normal",
        due_time: str = "Today",
        linked_entity: str = "[[Daily Priorities]]",
    ) -> Dict[str, Any]:
        prio = priority.lower().strip()
        if prio not in ("critical", "high", "normal", "low"):
            prio = "normal"
        now = datetime.now().isoformat(timespec="seconds")
        with self._get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO life_tasks (title, priority, status, due_time, linked_entity, created_at) VALUES (?, ?, 'pending', ?, ?, ?)",
                (title.strip(), prio, due_time, linked_entity, now),
            )
            task_id = cur.lastrowid
        return {"id": task_id, "title": title.strip(), "priority": prio, "status": "pending", "due_time": due_time}

    def toggle_task(self, task_id: Optional[int] = None, title_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if task_id is not None:
                row = conn.execute("SELECT id, title, status FROM life_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row and title_query:
                row = conn.execute(
                    "SELECT id, title, status FROM life_tasks WHERE lower(title) LIKE ? ORDER BY id DESC LIMIT 1",
                    (f"%{title_query.lower().strip()}%",),
                ).fetchone()
            if not row:
                return {"error": f"Task not found ({task_id or title_query})"}
            new_status = "completed" if row["status"] != "completed" else "pending"
            conn.execute("UPDATE life_tasks SET status = ? WHERE id = ?", (new_status, row["id"]))
        return {"id": row["id"], "title": row["title"], "status": new_status}

    def delete_task(self, task_id: Optional[int] = None, title_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if task_id is not None:
                row = conn.execute("SELECT id, title FROM life_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row and title_query:
                row = conn.execute(
                    "SELECT id, title FROM life_tasks WHERE lower(title) LIKE ? ORDER BY id DESC LIMIT 1",
                    (f"%{title_query.lower().strip()}%",),
                ).fetchone()
            if not row:
                return {"error": f"Task not found ({task_id or title_query})"}
            conn.execute("DELETE FROM life_tasks WHERE id = ?", (row["id"],))
        return {"deleted": row["id"], "title": row["title"]}

    def add_habit(self, name: str, target: str = "Daily") -> Dict[str, Any]:
        clean = name.strip()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO life_habits (name, streak, completed_today, target)
                VALUES (?, 1, 0, ?)
                ON CONFLICT(name) DO UPDATE SET target=excluded.target
                """,
                (clean, target),
            )
            row = conn.execute("SELECT * FROM life_habits WHERE name = ?", (clean,)).fetchone()
        return dict(row) if row else {"name": clean, "target": target}

    def toggle_habit(self, habit_id: Optional[int] = None, name_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if habit_id is not None:
                row = conn.execute(
                    "SELECT id, name, streak, completed_today FROM life_habits WHERE id = ?", (habit_id,)
                ).fetchone()
            if not row and name_query:
                row = conn.execute(
                    "SELECT id, name, streak, completed_today FROM life_habits WHERE lower(name) LIKE ? LIMIT 1",
                    (f"%{name_query.lower().strip()}%",),
                ).fetchone()
            if not row:
                return {"error": f"Habit not found ({habit_id or name_query})"}
            completed = 0 if row["completed_today"] else 1
            streak = row["streak"] + (1 if completed else -1)
            conn.execute(
                "UPDATE life_habits SET streak = ?, completed_today = ? WHERE id = ?",
                (max(0, streak), completed, row["id"]),
            )
        return {"id": row["id"], "name": row["name"], "completed_today": bool(completed), "streak": max(0, streak)}

    def delete_habit(self, habit_id: Optional[int] = None, name_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if habit_id is not None:
                row = conn.execute("SELECT id, name FROM life_habits WHERE id = ?", (habit_id,)).fetchone()
            if not row and name_query:
                row = conn.execute(
                    "SELECT id, name FROM life_habits WHERE lower(name) LIKE ? LIMIT 1",
                    (f"%{name_query.lower().strip()}%",),
                ).fetchone()
            if not row:
                return {"error": "Habit not found"}
            conn.execute("DELETE FROM life_habits WHERE id = ?", (row["id"],))
        return {"deleted": row["id"], "name": row["name"]}

    def add_schedule_event(
        self, time_slot: str, title: str, category: str = "Executive", set_active: bool = False
    ) -> Dict[str, Any]:
        with self._get_conn() as conn:
            if set_active:
                conn.execute("UPDATE life_schedule SET active = 0")
            cur = conn.execute(
                "INSERT INTO life_schedule (time_slot, title, category, active) VALUES (?, ?, ?, ?)",
                (time_slot.strip(), title.strip(), category.strip(), 1 if set_active else 0),
            )
            return {
                "id": cur.lastrowid,
                "time_slot": time_slot.strip(),
                "title": title.strip(),
                "category": category.strip(),
                "active": 1 if set_active else 0,
            }

    def set_active_schedule(self, event_id: Optional[int] = None, title_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if event_id is not None:
                row = conn.execute("SELECT id, title, time_slot FROM life_schedule WHERE id = ?", (event_id,)).fetchone()
            if not row and title_query:
                row = conn.execute(
                    "SELECT id, title, time_slot FROM life_schedule WHERE lower(title) LIKE ? OR time_slot LIKE ? LIMIT 1",
                    (f"%{title_query.lower().strip()}%", f"%{title_query.strip()}%"),
                ).fetchone()
            if not row:
                return {"error": "Schedule slot not found"}
            conn.execute("UPDATE life_schedule SET active = 0")
            conn.execute("UPDATE life_schedule SET active = 1 WHERE id = ?", (row["id"],))
        return {"id": row["id"], "title": row["title"], "time_slot": row["time_slot"], "active": 1}

    def delete_schedule_event(self, event_id: Optional[int] = None, title_query: Optional[str] = None) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = None
            if event_id is not None:
                row = conn.execute("SELECT id, title FROM life_schedule WHERE id = ?", (event_id,)).fetchone()
            if not row and title_query:
                row = conn.execute(
                    "SELECT id, title FROM life_schedule WHERE lower(title) LIKE ? OR time_slot LIKE ? LIMIT 1",
                    (f"%{title_query.lower().strip()}%", f"%{title_query.strip()}%"),
                ).fetchone()
            if not row:
                return {"error": "Schedule slot not found"}
            conn.execute("DELETE FROM life_schedule WHERE id = ?", (row["id"],))
        return {"deleted": row["id"], "title": row["title"]}

    def get_state(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            tasks = [dict(r) for r in conn.execute("SELECT * FROM life_tasks ORDER BY status DESC, id DESC").fetchall()]
            schedule = [dict(r) for r in conn.execute("SELECT * FROM life_schedule ORDER BY time_slot ASC").fetchall()]
            habits = [dict(r) for r in conn.execute("SELECT * FROM life_habits ORDER BY id ASC").fetchall()]
        return {
            "tasks": tasks,
            "schedule": schedule,
            "habits": habits,
        }


life_db = LifeCommandDB()
