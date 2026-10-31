"""MCP Tool Module: Life Command Matrix (Tasks, Schedule Timeline, Habit Streaks)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from server.core.life_db import life_db
from server.mcp_tools import register_mcp_tool



@register_mcp_tool(
    name="manage_tasks",
    description="Add, complete/toggle (by task_id or title), delete (by task_id or title), or list tasks in the Life Command Matrix.",
    category="Life Orchestration",
)
def manage_tasks(
    action: str = "list",
    title: Optional[str] = None,
    priority: str = "normal",
    due_time: str = "Today",
    task_id: Optional[int] = None,
) -> Dict[str, Any]:
    act = action.lower().strip()
    if act == "add" and title:
        created = life_db.add_task(title=title, priority=priority, due_time=due_time)
        return {
            "hud_action": "life_updated",
            "active_tab": "tasks",
            "created": created,
            "state": life_db.get_state(),
        }
    if act in ("complete", "toggle", "done"):
        toggled = life_db.toggle_task(task_id=task_id, title_query=title)
        return {
            "hud_action": "life_updated",
            "active_tab": "tasks",
            "toggled": toggled,
            "state": life_db.get_state(),
        }
    if act in ("delete", "remove"):
        deleted = life_db.delete_task(task_id=task_id, title_query=title)
        return {
            "hud_action": "life_updated",
            "active_tab": "tasks",
            "deleted": deleted,
            "state": life_db.get_state(),
        }
    return {
        "hud_action": "life_updated",
        "active_tab": "tasks",
        "state": life_db.get_state(),
    }


@register_mcp_tool(
    name="manage_schedule",
    description="Add a daily timeline slot, set an event as active now, delete an event, or list the daily schedule.",
    category="Life Orchestration",
)
def manage_schedule(
    action: str = "list",
    time_slot: str = "15:00",
    title: Optional[str] = None,
    category: str = "Executive",
    event_id: Optional[int] = None,
) -> Dict[str, Any]:
    act = action.lower().strip()
    if act == "add" and title:
        event = life_db.add_schedule_event(time_slot=time_slot, title=title, category=category)
        return {
            "hud_action": "life_updated",
            "active_tab": "schedule",
            "event": event,
            "state": life_db.get_state(),
        }
    if act in ("activate", "active", "start"):
        active_res = life_db.set_active_schedule(event_id=event_id, title_query=title or time_slot)
        return {
            "hud_action": "life_updated",
            "active_tab": "schedule",
            "activated": active_res,
            "state": life_db.get_state(),
        }
    if act in ("delete", "remove"):
        del_res = life_db.delete_schedule_event(event_id=event_id, title_query=title or time_slot)
        return {
            "hud_action": "life_updated",
            "active_tab": "schedule",
            "deleted": del_res,
            "state": life_db.get_state(),
        }
    return {
        "hud_action": "life_updated",
        "active_tab": "schedule",
        "state": life_db.get_state(),
    }


@register_mcp_tool(
    name="manage_habits",
    description="Add a new habit, complete/toggle a habit streak for today (by habit_id or name), delete a habit, or list habit streaks.",
    category="Life Orchestration",
)
def manage_habits(
    action: str = "list",
    name: Optional[str] = None,
    target: str = "Daily",
    habit_id: Optional[int] = None,
) -> Dict[str, Any]:
    act = action.lower().strip()
    if act == "add" and name:
        created = life_db.add_habit(name=name, target=target)
        return {
            "hud_action": "life_updated",
            "active_tab": "habits",
            "habit": created,
            "state": life_db.get_state(),
        }
    if act in ("complete", "toggle", "check", "done"):
        toggled = life_db.toggle_habit(habit_id=habit_id, name_query=name)
        return {
            "hud_action": "life_updated",
            "active_tab": "habits",
            "toggled": toggled,
            "state": life_db.get_state(),
        }
    if act in ("delete", "remove"):
        deleted = life_db.delete_habit(habit_id=habit_id, name_query=name)
        return {
            "hud_action": "life_updated",
            "active_tab": "habits",
            "deleted": deleted,
            "state": life_db.get_state(),
        }
    return {
        "hud_action": "life_updated",
        "active_tab": "habits",
        "state": life_db.get_state(),
    }


# Keep backward-compatible alias
@register_mcp_tool(
    name="schedule_event",
    description="Add a timeline block to the Life Command Matrix daily schedule.",
    category="Life Orchestration",
)
def schedule_event(time_slot: str, title: str, category: str = "Executive") -> Dict[str, Any]:
    return manage_schedule(action="add", time_slot=time_slot, title=title, category=category)
