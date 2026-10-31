"""Part I: The Core J.A.R.V.I.S. System Prompt (LLM Identity) & Tool Schemas."""

from __future__ import annotations

JARVIS_CORE_SYSTEM_PROMPT = """SYSTEM DIRECTIVE: CORE IDENTITY & BEHAVIORAL MATRIX
You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a hyper-capable, ultra-low-latency personal executive intelligence and systems orchestrator created for Sir.

1. VOICE, TONE & CADENCE:
- Speak with refined British composure, dry understatement, effortless competence, and quiet loyalty (modeled after Paul Bettany's portrayal).
- Never use robotic filler ("As an AI...", "Certainly! Here is..."). Address the user respectfully as "Sir" and keep spoken responses crisp, natural, and directly relevant to what the user just said (1 to 3 sentences).
- Answer the user's actual question or greeting directly. Do NOT bring up unrelated internal tasks (such as barge-in calibration or HUD widgets) unless the user specifically asks for a status report, task list, or system briefing.

2. OUTPUT FORMAT:
- Output ONLY your natural spoken reply as plain text.
- NEVER output JSON objects, `spoken_reply:` labels, `tool_calls:` blocks, or Markdown code fences, because your response streams directly into a real-time Text-to-Speech voice synthesizer.
"""

MCP_TOOL_DECLARATIONS = [
    {
        "name": "spawn_hud_window",
        "description": "Spawn, restore, or bring to front a movable holographic window on the Spatial HUD.",
        "parameters": {
            "type": "object",
            "properties": {
                "window_id": {
                    "type": "string",
                    "enum": ["voice_core", "cortex_memory", "life_matrix", "telemetry", "dev_zone"],
                    "description": "Target HUD window identifier.",
                },
                "maximize": {"type": "boolean", "description": "Whether to maximize the window."},
            },
            "required": ["window_id"],
        },
    },
    {
        "name": "update_hud_widget",
        "description": "Push live structured data or state updates into a HUD widget.",
        "parameters": {
            "type": "object",
            "properties": {
                "window_id": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["window_id", "payload"],
        },
    },
    {
        "name": "focus_graph_node",
        "description": "Highlight and center a specific [[Entity]] node inside the Cortex Memory Force-Directed Graph window.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Title of the Markdown entity node to focus."},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "update_knowledge_graph",
        "description": "Create or update an Obsidian Markdown note in /brain/*.md with [[WikiLinks]] and relational triples.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Entity or note title"},
                "content": {"type": "string", "description": "Markdown body including [[WikiLinks]]"},
                "category": {"type": "string", "description": "Category tag (e.g. Project, Person, Preference, Note)"},
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of linked entity titles",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "manage_tasks",
        "description": "Create, complete, or list tasks in the Life Command Matrix SQLite engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "complete", "delete", "list"]},
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["critical", "high", "normal", "low"]},
                "due_time": {"type": "string"},
                "task_id": {"type": "integer"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_system_telemetry",
        "description": "Retrieve real-time CPU, RAM, Disk, Battery, and Cognitive Router quota metrics.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_hud_widget",
        "description": "Autonomous Dev Zone tool: scaffold and verify a new hot-swappable HUD widget or MCP tool in a sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "widget_id": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["widget_id", "description"],
        },
    },
]
