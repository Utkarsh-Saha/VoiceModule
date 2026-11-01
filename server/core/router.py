"""Part III: Dynamic Hybrid Cognitive Router & Full-Feature AI Orchestrator.

Links every feature of Project J.A.R.V.I.S. directly to the AI:
1. Spatial Window Manager HUD (spawn, minimize, maximize, restore, close, open_all, minimize_all, reset_layout, switch_hud_tab)
2. Voice & Cognitive Router Settings (mute/unmute TTS, Echo Guard vs Headset mode, Auto/Cloud/Local router mode)
3. Cortex Memory Core (create/update Markdown [[WikiLink]] notes, delete notes, focus & open editor on 2D Force Graph nodes, 2-hop semantic RAG)
4. Life Command Matrix (add/complete/delete/list Tasks by title or ID, add/activate/delete Schedule timeline slots, add/complete/delete Habit streaks)
5. System Telemetry & Quota Monitor (real-time CPU, RAM, Disk, Battery, Network, RPM/TPM token buckets)
6. Autonomous Dev Zone (self-coding MCP tool & React HUD widget synthesizer with AST verification)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx

from server.config import ROOT_DIR, settings
from server.core.knowledge_graph import brain_engine
from server.core.life_db import life_db
from server.core.rate_limiter import TokenBucketRateLimiter
from server.mcp_tools import execute_mcp_tool, load_all_mcp_tools
from server.prompts import JARVIS_CORE_SYSTEM_PROMPT


AI_TOOL_DECLARATIONS = [
    {
        "name": "spawn_hud_window",
        "description": "Open, focus, or maximize a HUD window (voice_core, cortex_memory, life_matrix, telemetry, dev_zone).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "window_id": {"type": "STRING", "description": "voice_core, cortex_memory, life_matrix, telemetry, or dev_zone"},
                "maximize": {"type": "BOOLEAN"},
            },
            "required": ["window_id"],
        },
    },
    {
        "name": "control_hud_window",
        "description": "Minimize, maximize, restore, close, open_all, minimize_all, or reset_layout for HUD windows.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "minimize, maximize, restore, close, open_all, minimize_all, or reset_layout"},
                "window_id": {"type": "STRING", "description": "voice_core, cortex_memory, life_matrix, telemetry, dev_zone, or all"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "switch_hud_tab",
        "description": "Switch the active tab inside a HUD window (life_matrix: tasks/schedule/habits; cortex_memory: graph/editor; dev_zone: py/tsx/registry).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "window_id": {"type": "STRING"},
                "tab": {"type": "STRING"},
            },
            "required": ["window_id", "tab"],
        },
    },
    {
        "name": "control_voice_and_router",
        "description": "Change J.A.R.V.I.S. system settings: setting='tts' (value='mute'|'unmute'), setting='acoustic_mode' (value='echo_guard'|'headset'), or setting='router_mode' (value='auto'|'cloud'|'local').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "setting": {"type": "STRING"},
                "value": {"type": "STRING"},
            },
            "required": ["setting", "value"],
        },
    },
    {
        "name": "focus_graph_node",
        "description": "Highlight a node in the Cortex Memory Graph and optionally open its Markdown editor.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "node_id": {"type": "STRING"},
                "open_editor": {"type": "BOOLEAN"},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "update_knowledge_graph",
        "description": "Create or update an Obsidian Markdown note in /brain/*.md with [[WikiLinks]].",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "content": {"type": "STRING"},
                "category": {"type": "STRING"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "delete_knowledge_node",
        "description": "Delete a Markdown note from the Cortex Memory Core vault by title.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"title": {"type": "STRING"}},
            "required": ["title"],
        },
    },
    {
        "name": "manage_tasks",
        "description": "Add, complete, delete, or list tasks in the Life Command Matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "add, complete, delete, or list"},
                "title": {"type": "STRING"},
                "priority": {"type": "STRING", "description": "critical, high, normal, or low"},
                "due_time": {"type": "STRING"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "manage_schedule",
        "description": "Add, activate, delete, or list daily timeline events in the Life Command Matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "add, activate, delete, or list"},
                "time_slot": {"type": "STRING", "description": "24h time like 14:30 or 09:00"},
                "title": {"type": "STRING"},
                "category": {"type": "STRING"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "manage_habits",
        "description": "Add, complete/toggle, delete, or list daily habit streaks in the Life Command Matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "add, complete, delete, or list"},
                "name": {"type": "STRING"},
                "target": {"type": "STRING"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_system_telemetry",
        "description": "Fetch live CPU, RAM, Disk, Battery, Network, and Cognitive Router quota stats.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "generate_hud_widget",
        "description": "Synthesize, AST-verify, and mount a new MCP backend tool and React HUD widget in the Dev Zone.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "widget_name": {"type": "STRING"},
                "description": {"type": "STRING"},
            },
            "required": ["widget_name", "description"],
        },
    },
]


class HybridCognitiveRouter:
    """Routes user utterances across Cloud Gemini, Local Ollama, and Built-in Executive Engine."""

    def __init__(self) -> None:
        self.rate_limiter = TokenBucketRateLimiter(
            rpm_limit=settings.GEMINI_RPM_LIMIT,
            tpm_limit=settings.GEMINI_TPM_LIMIT,
        )
        self.ollama_online: bool = False
        self._ollama_warmed: bool = False
        self._last_ollama_check: float = 0.0
        self.preferred_mode: str = "local"  # Default to local Ollama GPU router ("local", "auto", "cloud")
        self._http_client: Optional[httpx.AsyncClient] = None
        load_all_mcp_tools()

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=1.5))
        return self._http_client

    def _get_live_gemini_key(self) -> str:
        load_dotenv(ROOT_DIR / ".env", override=True)
        key = os.getenv("GEMINI_API_KEY", "").strip()
        settings.GEMINI_API_KEY = key
        return key

    async def check_ollama_health(self, force: bool = False) -> bool:
        now = time.time()
        if self.ollama_online and (now - self._last_ollama_check < 60.0):
            return True
        self._last_ollama_check = now
        base_url = settings.OLLAMA_BASE_URL.replace("localhost", "127.0.0.1")
        client = self._get_http_client()
        try:
            resp = await client.get(f"{base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                self.ollama_online = True
                if not self._ollama_warmed:
                    self._ollama_warmed = True
                    asyncio.create_task(self._warmup_ollama_vram(base_url))
                return True
        except Exception:
            self.ollama_online = False

        # Auto-start `ollama serve` in background if installed on Windows/host
        if self.preferred_mode == "local" or force:
            try:
                import shutil
                import subprocess

                ollama_bin = shutil.which("ollama") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
                if os.path.exists(ollama_bin):
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    subprocess.Popen(
                        [ollama_bin, "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=flags,
                    )
                    await asyncio.sleep(2.0)
                    resp = await client.get(f"{base_url}/api/tags", timeout=2.0)
                    self.ollama_online = resp.status_code == 200
                    if self.ollama_online and not self._ollama_warmed:
                        self._ollama_warmed = True
                        asyncio.create_task(self._warmup_ollama_vram(base_url))
            except Exception:
                self.ollama_online = False

        return self.ollama_online

    async def _warmup_ollama_vram(self, base_url: str) -> None:
        """Pins `jarvis:latest` 100% into GPU VRAM (`keep_alive=60m`, `num_ctx=1024`) so first turn has zero cold-load latency."""
        try:
            client = self._get_http_client()
            await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": "jarvis:latest",
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_ctx": 1024, "num_predict": 1},
                    "keep_alive": "60m",
                    "stream": False,
                },
                timeout=15.0,
            )
        except Exception:
            pass

    @staticmethod
    def _detect_target_window(low: str) -> Optional[str]:
        if any(w in low for w in ["cortex", "brain", "graph", "memory", "obsidian", "knowledge"]):
            return "cortex_memory"
        if any(w in low for w in ["task", "schedule", "calendar", "habit", "life", "matrix", "todo", "directive"]):
            return "life_matrix"
        if any(w in low for w in ["telemetry", "cpu", "ram", "quota", "router", "monitor", "system"]):
            return "telemetry"
        if any(w in low for w in ["dev zone", "devzone", "sandbox", "code", "widget", "registry"]):
            return "dev_zone"
        if any(w in low for w in ["voice", "arc", "reactor", "audio", "mic"]):
            return "voice_core"
        return None

    def _extract_autonomous_tool_calls(self, user_text: str) -> List[Dict[str, Any]]:
        """Comprehensive fast-path intent parser linking all HUD, Voice, Router, Brain, Life, Telemetry, and DevZone features."""
        low = user_text.lower().strip()
        calls: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 1. Global HUD Window Controls (Reset, Open All, Minimize All, Close, Minimize, Maximize)
        # ------------------------------------------------------------------
        if any(p in low for p in ["reset hud", "reset layout", "default layout", "restore layout", "reset windows"]):
            calls.append({"name": "control_hud_window", "arguments": {"action": "reset_layout", "window_id": "all"}})
        elif any(p in low for p in ["minimize all", "hide all windows", "clear desktop", "collapse all"]):
            calls.append({"name": "control_hud_window", "arguments": {"action": "minimize_all", "window_id": "all"}})
        elif any(p in low for p in ["open all windows", "show all windows", "restore all windows", "bring up everything"]):
            calls.append({"name": "control_hud_window", "arguments": {"action": "open_all", "window_id": "all"}})
        else:
            target_win = self._detect_target_window(low)
            if target_win:
                if any(k in low for k in ["minimize", "collapse", "hide"]):
                    calls.append({"name": "control_hud_window", "arguments": {"action": "minimize", "window_id": target_win}})
                elif any(k in low for k in ["close", "shut", "dismiss"]):
                    calls.append({"name": "control_hud_window", "arguments": {"action": "close", "window_id": target_win}})
                elif any(k in low for k in ["restore", "unmaximize", "normal size"]):
                    calls.append({"name": "control_hud_window", "arguments": {"action": "restore", "window_id": target_win}})
                elif any(k in low for k in ["maximize", "full screen", "fullscreen", "expand"]):
                    calls.append({"name": "control_hud_window", "arguments": {"action": "maximize", "window_id": target_win}})
                elif any(k in low for k in ["open", "spawn", "show", "bring up", "display", "focus"]):
                    calls.append({"name": "spawn_hud_window", "arguments": {"window_id": target_win, "maximize": False}})

        # ------------------------------------------------------------------
        # 2. HUD Inner Tab Switching (Tasks / Schedule / Habits / Editor / Registry)
        # ------------------------------------------------------------------
        if any(p in low for p in ["show schedule", "open schedule", "view timeline", "my schedule", "daily timeline", "calendar", "schedule tab", "timeline tab"]):
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "schedule"}})
        elif any(p in low for p in ["show habits", "open habits", "habit streaks", "my habits", "view streaks", "habits tab", "habit tab"]):
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "habits"}})
        elif any(p in low for p in ["tasks tab", "directives tab", "show tasks", "open tasks"]):
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "tasks"}})
        elif any(p in low for p in ["open markdown editor", "open note editor", "edit markdown", "show editor", "editor tab"]):
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "cortex_memory", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "cortex_memory", "tab": "editor"}})
        elif any(p in low for p in ["show tool registry", "open registry", "list mcp tools", "view registry", "registry tab"]):
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "dev_zone", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "dev_zone", "tab": "registry"}})

        # ------------------------------------------------------------------
        # 3. Voice, Acoustic Mode & Cognitive Router Controls
        # ------------------------------------------------------------------
        if any(p in low for p in ["mute voice", "mute tts", "stop speaking", "be quiet", "silence voice"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "tts", "value": "mute"}})
        elif any(p in low for p in ["unmute voice", "unmute tts", "enable voice", "turn on voice"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "tts", "value": "unmute"}})

        if any(p in low for p in ["headset mode", "headphone mode", "bluetooth mode", "earbuds mode"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "acoustic_mode", "value": "headset"}})
        elif any(p in low for p in ["echo guard", "speaker mode", "open speaker", "anti-echo", "anti echo"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "acoustic_mode", "value": "echo_guard"}})

        if any(p in low for p in ["switch to local", "use local model", "use ollama", "offline mode", "local router"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "router_mode", "value": "local"}})
        elif any(p in low for p in ["switch to cloud", "use gemini", "cloud mode", "cloud router"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "router_mode", "value": "cloud"}})
        elif any(p in low for p in ["switch to auto", "auto router", "automatic routing"]):
            calls.append({"name": "control_voice_and_router", "arguments": {"setting": "router_mode", "value": "auto"}})

        # ------------------------------------------------------------------
        # 4. Live System Telemetry & Diagnostics
        # ------------------------------------------------------------------
        is_dev_or_sched = any(w in low for w in ["build ", "generate ", "create a new ", "schedule "])
        if not is_dev_or_sched and any(
            w in low for w in ["status report", "system telemetry", "system status", "cpu", "ram", "memory usage", "system diagnostic", "briefing", "battery"]
        ):
            calls.append({"name": "get_system_telemetry", "arguments": {}})
            calls.append({"name": "manage_tasks", "arguments": {"action": "list"}})

        # ------------------------------------------------------------------
        # 5. Task Management (Add, Complete, Delete, List)
        # ------------------------------------------------------------------
        task_add = re.search(
            r"(?:add|create|new|log)\s+(?:a\s+)?(?:(critical|high|normal|low)\s+(?:priority\s+)?)?(?:task|directive|todo)[:\s]+(.+)",
            user_text,
            re.IGNORECASE,
        )
        if task_add:
            prio = (task_add.group(1) or "high").lower()
            title = task_add.group(2).strip().rstrip(".")
            calls.append({"name": "manage_tasks", "arguments": {"action": "add", "title": title, "priority": prio}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})

        task_done = re.search(
            r"(?:(?:complete|finish|check\s+off|toggle)\s+(?:the\s+)?task[:\s]+(.+)|mark\s+(?:the\s+)?(?:task\s+)?(.+?)\s+as\s+(?:done|complete|completed|finished))",
            user_text,
            re.IGNORECASE,
        )
        if task_done:
            t_query = (task_done.group(1) or task_done.group(2) or "").strip().rstrip(".")
            calls.append({"name": "manage_tasks", "arguments": {"action": "complete", "title": t_query}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})

        task_del = re.search(
            r"(?:delete|remove)\s+(?:the\s+)?task[:\s]+(.+)",
            user_text,
            re.IGNORECASE,
        )
        if task_del:
            t_query = task_del.group(1).strip().rstrip(".")
            calls.append({"name": "manage_tasks", "arguments": {"action": "delete", "title": t_query}})

        # ------------------------------------------------------------------
        # 6. Schedule / Timeline Management (Add, Activate, Delete)
        # ------------------------------------------------------------------
        sched_add_1 = re.search(
            r"(?:schedule|add\s+(?:to\s+)?schedule|add\s+event)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(.+)",
            user_text,
            re.IGNORECASE,
        )
        sched_add_2 = re.search(
            r"(?:schedule|add\s+(?:to\s+)?schedule|add\s+event)\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            user_text,
            re.IGNORECASE,
        )
        if sched_add_1:
            ev_time = sched_add_1.group(1).strip()
            ev_title = sched_add_1.group(2).strip().rstrip(".")
            calls.append({"name": "manage_schedule", "arguments": {"action": "add", "time_slot": ev_time, "title": ev_title}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "schedule"}})
        elif sched_add_2:
            ev_title = sched_add_2.group(1).strip().rstrip(".")
            ev_time = sched_add_2.group(2).strip()
            calls.append({"name": "manage_schedule", "arguments": {"action": "add", "time_slot": ev_time, "title": ev_title}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "schedule"}})

        # ------------------------------------------------------------------
        # 7. Habit Streaks (Add, Complete/Toggle, Delete)
        # ------------------------------------------------------------------
        habit_add = re.search(
            r"(?:add|create|start|new)\s+(?:a\s+)?habit[:\s]+(.+)",
            user_text,
            re.IGNORECASE,
        )
        if habit_add:
            h_name = habit_add.group(1).strip().rstrip(".")
            calls.append({"name": "manage_habits", "arguments": {"action": "add", "name": h_name}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "habits"}})

        habit_done = re.search(
            r"(?:complete|check\s+off|log|toggle|finish)\s+(?:the\s+)?habit[:\s]+(.+)",
            user_text,
            re.IGNORECASE,
        )
        if habit_done:
            h_name = habit_done.group(1).strip().rstrip(".")
            calls.append({"name": "manage_habits", "arguments": {"action": "complete", "name": h_name}})
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "life_matrix", "maximize": False}})
            calls.append({"name": "switch_hud_tab", "arguments": {"window_id": "life_matrix", "tab": "habits"}})

        # ------------------------------------------------------------------
        # 8. Cortex Memory Core (Delete Note, Create/Update Note, Focus Node)
        # ------------------------------------------------------------------
        del_note_match = re.search(
            r"(?:delete|remove)\s+(?:the\s+)?(?:note|node|memory)\s+(?:about|titled|called)?\s*(.+)",
            user_text,
            re.IGNORECASE,
        )
        if del_note_match:
            n_title = del_note_match.group(1).strip().rstrip(".")
            calls.append({"name": "delete_knowledge_node", "arguments": {"title": n_title}})
        else:
            mem_match = re.search(
                r"^(?:please\s+|jarvis\s+)?(?:remember|note\s+that|create\s+(?:a\s+)?note|record\s+that|save\s+to\s+(?:cortex|brain|memory|vault))\s+(?:that\s+)?(.+)",
                user_text.strip(),
                re.IGNORECASE,
            )
            if mem_match:
                fact = mem_match.group(1).strip()
                words = [w.capitalize() for w in re.findall(r"[a-zA-Z0-9]+", fact)[:3]]
                title = " ".join(words) if words else "Executive Note"
                calls.append(
                    {
                        "name": "update_knowledge_graph",
                        "arguments": {
                            "title": title,
                            "content": f"Recorded directive from [[Sir]]: {fact}\n\nLinked to [[Project J.A.R.V.I.S.]] and [[Daily Priorities]].",
                            "category": "Directive",
                            "links": ["Sir", "Project J.A.R.V.I.S.", "Daily Priorities"],
                        },
                    }
                )
                calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "cortex_memory", "maximize": False}})

            for node_id in list(brain_engine.graph.nodes):
                if node_id.lower() != "sir" and len(node_id) > 3 and node_id.lower() in low:
                    open_ed = any(w in low for w in ["edit", "editor", "markdown", "note"])
                    calls.append({"name": "focus_graph_node", "arguments": {"node_id": node_id, "open_editor": open_ed}})
                    break

        # ------------------------------------------------------------------
        # 9. Autonomous Dev Zone Self-Coding
        # ------------------------------------------------------------------
        dev_match = re.search(
            r"(?:build|create|generate|code)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?([a-zA-Z0-9 _-]{3,30}?)\s+(?:window|widget|module|tracker|tool)",
            user_text,
            re.IGNORECASE,
        )
        if dev_match:
            w_name = dev_match.group(1).strip()
            calls.append(
                {
                    "name": "generate_hud_widget",
                    "arguments": {
                        "widget_name": w_name,
                        "description": f"Autonomous HUD module for {w_name} requested by Sir.",
                    },
                }
            )
            calls.append({"name": "spawn_hud_window", "arguments": {"window_id": "dev_zone", "maximize": False}})

        return calls

    @staticmethod
    def _is_followup_utterance(user_text: str) -> bool:
        """Returns True only when the user's message explicitly refers to the immediately preceding turn."""
        low = user_text.lower().strip()
        words = set(re.findall(r"[a-z]+", low))
        followup_pronouns = {"it", "that", "this", "those", "them", "they", "he", "she", "there"}
        followup_starters = (
            "and ",
            "also ",
            "what about",
            "how about",
            "why ",
            "how ",
            "elaborate",
            "explain that",
            "do it",
            "yes",
            "no",
            "sure",
            "go ahead",
        )
        if words & followup_pronouns:
            return True
        if low.startswith(followup_starters):
            return True
        return False

    def _build_contextual_prompt(
        self,
        user_text: str,
        rag_context: str,
        executed_tools: List[Dict[str, Any]],
        recent_turns: List[Dict[str, str]],
    ) -> str:
        sections: List[str] = []
        low = user_text.lower().strip()

        # Only inject live HUD state when the user actually asks about their tasks, schedule, habits, or status
        asks_about_life_or_state = any(
            w in low
            for w in [
                "task",
                "directive",
                "todo",
                "schedule",
                "timeline",
                "calendar",
                "agenda",
                "habit",
                "streak",
                "pending",
            ]
        )
        if asks_about_life_or_state and not executed_tools:
            life = life_db.get_state()
            pending_tasks = [f"{t['title']} ({t['priority']})" for t in life["tasks"] if t["status"] != "completed"]
            active_habits = [f"{h['name']} ({h['streak']}d streak)" for h in life["habits"]]
            schedule_slots = [f"{s['time_slot']} {s['title']}" for s in life["schedule"]]
            sections.append(
                "[LIVE SYSTEM & HUD STATE]\n"
                f"- Pending Directives ({len(pending_tasks)}): {', '.join(pending_tasks[:5]) or 'None'}\n"
                f"- Daily Schedule: {', '.join(schedule_slots[:4])}\n"
                f"- Active Habits: {', '.join(active_habits[:4])}"
            )

        # Only include recent conversation if this utterance is a genuine follow-up AND no standalone tool just ran
        if recent_turns and not executed_tools and self._is_followup_utterance(user_text):
            history_lines = [
                f"{'Sir' if t['role'] == 'user' else 'J.A.R.V.I.S.'}: {t['text']}"
                for t in recent_turns[-2:]
            ]
            sections.append("[IMMEDIATE PREVIOUS EXCHANGE]\n" + "\n".join(history_lines))

        if executed_tools:
            tool_summaries = []
            for t in executed_tools:
                res = dict(t.get("result") or {})
                # Strip bulky full-state arrays so the LLM focuses strictly on the exact action taken
                res.pop("state", None)
                res.pop("graph", None)
                res.pop("generated_py", None)
                res.pop("generated_tsx", None)
                tool_summaries.append(f"- JUST EXECUTED `{t['tool']}` -> {json.dumps(res)}")
            sections.append(
                "[ACTIONS JUST EXECUTED ON THIS TURN (CONFIRM ONLY THESE ACTIONS TO SIR — DO NOT INVENT EXTRA DETAILS)]\n"
                + "\n".join(tool_summaries)
            )

        if rag_context.strip() and not executed_tools:
            sections.append(f"[RELEVANT CORTEX MEMORY VAULT NOTES]\n{rag_context.strip()}")

        if not sections:
            return user_text

        sections.append(
            f"[CURRENT MESSAGE FROM SIR]\n{user_text}\n\n"
            f"Respond directly and accurately to Sir's current message in 1 to 2 crisp, natural spoken sentences. "
            f"Do not mention unrelated tasks, schedule items, or past turns unless Sir explicitly asked about them."
        )
        return "\n\n".join(sections)

    async def _call_gemini_cloud(
        self,
        full_prompt: str,
        executed_tools: List[Dict[str, Any]],
    ) -> Optional[tuple[str, str]]:
        api_key = self._get_live_gemini_key()
        if not api_key:
            return None
        if not self.rate_limiter.can_use_cloud(estimated_tokens=len(full_prompt) // 3 + 300):
            return None

        from google import genai
        from google.genai import types

        start = time.perf_counter()
        candidate_models = list(
            dict.fromkeys(
                [
                    os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.1-flash-lite").strip(),
                    "gemini-3.1-flash-lite",
                    "gemini-3.5-flash-lite",
                    "gemini-flash-lite-latest",
                ]
            )
        )

        already_called = {t["tool"] for t in executed_tools}

        for model_name in candidate_models:
            try:
                client = genai.Client(api_key=api_key)
                # Only attach function_declarations when fast-path hasn't already executed the tools
                cfg = types.GenerateContentConfig(
                    system_instruction=JARVIS_CORE_SYSTEM_PROMPT,
                    temperature=0.35,
                    max_output_tokens=300,
                    tools=[{"function_declarations": AI_TOOL_DECLARATIONS}] if not executed_tools else None,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                    if "3.1" in model_name
                    else None,
                )
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                        config=cfg,
                    ),
                    timeout=9.0,
                )

                # Execute any native function calls emitted by Gemini that haven't already run
                fn_calls = getattr(response, "function_calls", None) or []
                for fc in fn_calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args) if fc.args else {}
                    if fn_name not in already_called:
                        exec_res = execute_mcp_tool(fn_name, fn_args)
                        executed_tools.append(exec_res)
                        already_called.add(fn_name)
                        res_payload = exec_res.get("result") or {}
                        if res_payload.get("hud_action") == "system_setting" and res_payload.get("setting") == "router_mode":
                            val = res_payload.get("value")
                            if val in ("auto", "cloud", "local"):
                                self.preferred_mode = val

                # Extract text parts without triggering google-genai non-text warning
                text_parts: List[str] = []
                for cand in getattr(response, "candidates", None) or []:
                    content = getattr(cand, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        if getattr(part, "text", None):
                            text_parts.append(part.text)
                text = " ".join(text_parts).strip()

                if not text and executed_tools:
                    # Synthesize natural spoken confirmation for the invoked tool(s)
                    text = self._synthesize_executive_reply("", {"primary_nodes": []}, executed_tools)

                if text:
                    text = self._clean_spoken_text(text)
                    latency = (time.perf_counter() - start) * 1000.0
                    est_tokens = len(full_prompt) // 4 + len(text) // 4
                    self.rate_limiter.record_cloud_usage(tokens_used=est_tokens, latency_ms=latency)
                    return text, model_name
            except Exception:
                continue
        return None

    async def _call_ollama_local(self, full_prompt: str) -> Optional[tuple[str, str]]:
        if not await self.check_ollama_health(force=False):
            return None

        start = time.perf_counter()
        base_url = settings.OLLAMA_BASE_URL.replace("localhost", "127.0.0.1")
        candidate_local = list(
            dict.fromkeys(["jarvis:latest", "llama3.2:3b", settings.OLLAMA_MODEL])
        )
        client = self._get_http_client()

        for model_name in candidate_local:
            try:
                # jarvis:latest already has the J.A.R.V.I.S. SYSTEM prompt compiled into its Modelfile
                if model_name == "jarvis:latest":
                    messages = [{"role": "user", "content": full_prompt}]
                else:
                    messages = [
                        {"role": "system", "content": JARVIS_CORE_SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt},
                    ]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "options": {
                        "temperature": 0.35,
                        "num_ctx": 1024,
                        "num_predict": 60,
                    },
                    "keep_alive": "60m",
                    "stream": False,
                }
                resp = await client.post(f"{base_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        cleaned = self._clean_spoken_text(content)
                        latency = (time.perf_counter() - start) * 1000.0
                        self.rate_limiter.record_local_usage(latency_ms=latency, tier_name="ollama-local")
                        self.ollama_online = True
                        return cleaned, model_name
            except Exception:
                self.ollama_online = False
                continue
        return None

    @staticmethod
    def _clean_spoken_text(raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        if cleaned.startswith("{") and "spoken_reply" in cleaned:
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "spoken_reply" in parsed:
                    return str(parsed["spoken_reply"]).strip()
            except Exception:
                pass

        cleaned = re.sub(r"\n*\{[\s\S]*?\"spoken_reply\"[\s\S]*?\}\s*$", "", cleaned).strip()
        cleaned = re.sub(r"\n*tool_calls\s*:\s*\[[\s\S]*$", "", cleaned, flags=re.IGNORECASE).strip()

        m = re.match(r'^spoken_reply\s*:\s*"(.+?)"\s*$', cleaned, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
        else:
            cleaned = re.sub(r'^spoken_reply\s*:\s*"?', "", cleaned, flags=re.IGNORECASE).rstrip('"').strip()

        return cleaned or raw.strip()

    def _synthesize_executive_reply(
        self,
        user_text: str,
        rag_result: Dict[str, Any],
        executed_tools: List[Dict[str, Any]],
    ) -> str:
        low = user_text.lower().strip()

        for tool_exec in executed_tools:
            t_name = tool_exec["tool"]
            res = tool_exec.get("result", {})
            if t_name == "generate_hud_widget":
                w_name = res.get("widget_name", "Custom")
                return (
                    f"Right away, Sir. I have branched the Dev Zone sandbox, synthesized the {w_name} "
                    f"MCP backend tool and HUD component, verified its AST syntax, and mounted it to your active registry."
                )
            if t_name == "manage_tasks":
                if "created" in res:
                    return f"Understood, Sir. I have logged '{res['created']['title']}' as a {res['created']['priority']}-priority directive inside the Life Command Matrix."
                if "toggled" in res and "title" in res["toggled"]:
                    return f"Consider it done, Sir. I have marked '{res['toggled']['title']}' as {res['toggled']['status']} in the Life Command Matrix."
                if "deleted" in res and "title" in res["deleted"]:
                    return f"Removed '{res['deleted']['title']}' from your active directives, Sir."
            if t_name == "manage_schedule":
                if "event" in res:
                    return f"Scheduled '{res['event']['title']}' at {res['event']['time_slot']} on your timeline, Sir."
                if "activated" in res:
                    return f"Activated '{res['activated'].get('title', 'the timeline block')}' on your schedule now, Sir."
            if t_name == "manage_habits":
                if "habit" in res:
                    return f"Added '{res['habit']['name']}' to your daily habit tracker, Sir."
                if "toggled" in res and "name" in res["toggled"]:
                    return f"Logged '{res['toggled']['name']}' for today, Sir. Your streak now stands at {res['toggled']['streak']} days."
            if t_name == "update_knowledge_graph":
                node_id = res.get("node_id", "Memory Node")
                return f"Noted, Sir. I have committed that to the Cortex Memory Core vault under '{node_id}'."
            if t_name == "delete_knowledge_node":
                return "I have purged that note from the Cortex Memory Core and re-indexed the graph, Sir."
            if t_name == "control_hud_window":
                ctrl = res.get("control", "updated")
                win_title = res.get("title", "HUD window")
                return f"Very good, Sir. I have applied '{ctrl}' to {win_title} on your Spatial HUD."
            if t_name == "control_voice_and_router":
                return f"Updated {res.get('setting')} to {res.get('value')}, Sir."
            if t_name == "switch_hud_tab":
                return f"Switched the {res.get('window_id')} view to the {res.get('tab')} panel, Sir."
            if t_name == "spawn_hud_window":
                win_title = res.get("title", "HUD Window")
                return f"Bringing the {win_title} online and centering it on your viewport now, Sir."
            if t_name == "get_system_telemetry":
                tasks = execute_mcp_tool("manage_tasks", {"action": "list"})["result"]["state"]["tasks"]
                pending = sum(1 for t in tasks if t["status"] != "completed")
                return (
                    f"All systems are operating at peak efficiency, Sir. Host CPU load is at {res.get('cpu_percent')}% "
                    f"with {res.get('ram_used_gb')} GB of RAM committed, and you have {pending} active directives in the Life Command Matrix."
                )

        if any(w in low for w in ["audible", "audibles", "hear me", "mic check", "testing", "texting", "working"]):
            return (
                "Loud and clear, Sir. Both your text and full-duplex audio channels are locked in and operating "
                "at nominal parameters."
            )

        if low in {"hi", "hey", "hello", "good morning", "good afternoon", "good evening", "yo", "sup"} or any(
            w in low for w in ["who are you", "introduce", "wake up", "hey jarvis", "hello jarvis"]
        ):
            return (
                "At your service, Sir. All HUD subsystems and the Cortex Memory Core are online. "
                "How may I assist you?"
            )

        primary = rag_result.get("primary_nodes", [])
        if primary:
            lead = primary[0]
            return (
                f"I have pulled up the '{lead}' node inside your Cortex Memory Core, Sir. "
                f"Would you like me to update its Markdown notes or log a linked action item?"
            )

        return (
            f"Understood, Sir. I have processed your request. "
            "Let me know if you would like me to update your Life Command Matrix, Cortex Memory Graph, or Spatial HUD."
        )

    async def route_and_respond(self, user_text: str) -> Dict[str, Any]:
        overall_start = time.perf_counter()
        recent_turns = brain_engine.get_recent_turns(limit=4)
        brain_engine.log_conversation_turn("user", user_text)

        # 1. Retrieve Semantic + 2-Hop Graph Context (only if genuinely relevant)
        rag_result = brain_engine.retrieve_context(user_text, top_k=2)

        # 2. Detect & Execute Fast-Path Autonomous MCP Tool Calls
        planned_tools = self._extract_autonomous_tool_calls(user_text)
        executed_tools: List[Dict[str, Any]] = []
        for call in planned_tools:
            exec_res = execute_mcp_tool(call["name"], call["arguments"])
            executed_tools.append(exec_res)
            res_payload = exec_res.get("result") or {}
            if res_payload.get("hud_action") == "system_setting" and res_payload.get("setting") == "router_mode":
                val = res_payload.get("value")
                if val in ("auto", "cloud", "local"):
                    self.preferred_mode = val

        full_prompt = self._build_contextual_prompt(
            user_text=user_text,
            rag_context=rag_result["context_block"],
            executed_tools=executed_tools,
            recent_turns=recent_turns,
        )

        # 3. Arbitrate LLM Tier (with native Function Calling enabled on Gemini)
        spoken_reply: Optional[str] = None
        tier_used = "executive-local-core"

        # Fast-path <1ms deterministic confirmation when local MCP tools have already executed the directive
        if self.preferred_mode == "local" and executed_tools:
            spoken_reply = self._synthesize_executive_reply(user_text, rag_result, executed_tools)
            latency = (time.perf_counter() - overall_start) * 1000.0
            self.rate_limiter.record_local_usage(latency_ms=latency, tier_name="executive-local-core")

        if not spoken_reply and self.preferred_mode != "local":
            cloud_res = await self._call_gemini_cloud(full_prompt, executed_tools)
            if cloud_res:
                spoken_reply, model_used = cloud_res
                tier_used = f"gemini-cloud ({model_used})"

        if not spoken_reply:
            ollama_res = await self._call_ollama_local(full_prompt)
            if ollama_res:
                spoken_reply, ollama_model = ollama_res
                tier_used = f"ollama-local ({ollama_model})"

        if not spoken_reply:
            latency = (time.perf_counter() - overall_start) * 1000.0
            self.rate_limiter.record_local_usage(latency_ms=latency, tier_name="executive-local-core")
            spoken_reply = self._synthesize_executive_reply(user_text, rag_result, executed_tools)

        brain_engine.log_conversation_turn("jarvis", spoken_reply)
        total_ms = round((time.perf_counter() - overall_start) * 1000.0, 1)

        return {
            "spoken_reply": spoken_reply,
            "tier_used": tier_used,
            "latency_ms": total_ms,
            "rag": {
                "primary_nodes": rag_result["primary_nodes"],
                "two_hop_neighbors": rag_result["two_hop_neighbors"],
                "highlighted_nodes": brain_engine.active_highlighted_nodes,
            },
            "tool_executions": executed_tools,
            "router_metrics": self.rate_limiter.snapshot(),
        }

    async def shutdown(self) -> None:
        """Close shared HTTP client connections and free resources."""
        if self._http_client is not None and not self._http_client.is_closed:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None


cognitive_router = HybridCognitiveRouter()
