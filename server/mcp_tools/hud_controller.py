"""MCP Tool Module: Spatial HUD Window, Tab, Voice & Router Controller."""

from __future__ import annotations

from typing import Any, Dict, Optional

from server.mcp_tools import register_mcp_tool

VALID_WINDOWS = {
    "voice_core": "Central Arc Voice Core",
    "cortex_memory": "Cortex Memory Core (Cortex Graph)",
    "life_matrix": "Life Command Matrix",
    "telemetry": "MCP Telemetry & Router Monitor",
    "dev_zone": "Autonomous Dev Zone Sandbox",
}


def _normalize_window_id(raw_id: str) -> str:
    low = (raw_id or "").lower().strip()
    if low in VALID_WINDOWS:
        return low
    if any(w in low for w in ["cortex", "brain", "graph", "memory", "obsidian"]):
        return "cortex_memory"
    if any(w in low for w in ["life", "task", "schedule", "calendar", "habit", "matrix"]):
        return "life_matrix"
    if any(w in low for w in ["telemetry", "cpu", "ram", "router", "quota", "monitor"]):
        return "telemetry"
    if any(w in low for w in ["dev", "sandbox", "code", "widget"]):
        return "dev_zone"
    if any(w in low for w in ["voice", "arc", "audio", "mic"]):
        return "voice_core"
    return "cortex_memory"


@register_mcp_tool(
    name="spawn_hud_window",
    description="Spawn, restore, or focus a movable holographic window on the Spatial HUD.",
    category="Spatial HUD",
)
def spawn_hud_window(window_id: str, maximize: bool = False) -> Dict[str, Any]:
    normalized = _normalize_window_id(window_id)
    return {
        "hud_action": "spawn_window",
        "window_id": normalized,
        "title": VALID_WINDOWS.get(normalized, window_id),
        "maximize": bool(maximize),
    }


@register_mcp_tool(
    name="control_hud_window",
    description="Minimize, maximize, restore, close, open_all, minimize_all, or reset_layout for HUD windows.",
    category="Spatial HUD",
)
def control_hud_window(action: str, window_id: str = "all") -> Dict[str, Any]:
    act = action.lower().strip()
    normalized = "all" if window_id.lower().strip() == "all" else _normalize_window_id(window_id)
    return {
        "hud_action": "window_control",
        "control": act,
        "window_id": normalized,
        "title": VALID_WINDOWS.get(normalized, "All HUD Windows"),
    }


@register_mcp_tool(
    name="switch_hud_tab",
    description="Switch active view/tab inside a HUD window (e.g. life_matrix: tasks/schedule/habits; cortex_memory: graph/editor; dev_zone: py/tsx/registry).",
    category="Spatial HUD",
)
def switch_hud_tab(window_id: str, tab: str) -> Dict[str, Any]:
    normalized = _normalize_window_id(window_id)
    return {
        "hud_action": "switch_tab",
        "window_id": normalized,
        "tab": tab.lower().strip(),
    }


@register_mcp_tool(
    name="control_voice_and_router",
    description="Control J.A.R.V.I.S. voice TTS (mute/unmute), acoustic mode (echo_guard/headset), or cognitive router tier (auto/cloud/local).",
    category="System Control",
)
def control_voice_and_router(setting: str, value: str) -> Dict[str, Any]:
    from server.core.router import cognitive_router

    s_clean = setting.lower().strip()
    v_clean = value.lower().strip()

    if s_clean == "router_mode" and v_clean in ("auto", "cloud", "local"):
        cognitive_router.preferred_mode = v_clean

    return {
        "hud_action": "system_setting",
        "setting": s_clean,
        "value": v_clean,
    }


@register_mcp_tool(
    name="update_hud_widget",
    description="Push live structured data or visual filter state into an active HUD widget.",
    category="Spatial HUD",
)
def update_hud_widget(window_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "hud_action": "update_widget",
        "window_id": _normalize_window_id(window_id),
        "payload": payload or {},
    }


@register_mcp_tool(
    name="focus_graph_node",
    description="Center and illuminate a specific [[Entity]] node inside the Cortex Memory Graph, optionally opening its Markdown editor.",
    category="Spatial HUD",
)
def focus_graph_node(node_id: str, open_editor: bool = False) -> Dict[str, Any]:
    from server.core.knowledge_graph import brain_engine

    clean_id = node_id.strip().strip("[]")
    brain_engine.active_highlighted_nodes = [clean_id]
    note = brain_engine.get_note(clean_id)
    return {
        "hud_action": "focus_graph_node",
        "node_id": clean_id,
        "open_editor": bool(open_editor),
        "note": note,
    }
