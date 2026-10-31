"""Project J.A.R.V.I.S. — FastAPI Hybrid Router + Full-Duplex WebSocket Spine.

Provides:
- Bidirectional `/ws/jarvis` WebSocket supporting simultaneous 16kHz PCM audio frames, JSON utterances, Neural British Male TTS audio streaming, and <180ms barge-in interruption.
- Local Faster-Whisper `/api/stt` endpoint & Neural `/api/tts` endpoint.
- REST endpoints for the Cortex Memory Core Graph, Life Command Matrix, System Telemetry, and Autonomous Dev Zone.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.config import settings
from server.core.knowledge_graph import brain_engine
from server.core.life_db import life_db
from server.core.router import cognitive_router
from server.core.voice_engine import voice_engine
from server.mcp_tools import execute_mcp_tool, get_mcp_registry_state
from server.mcp_tools.system_telemetry import get_system_telemetry

app = FastAPI(
    title="PROJECT J.A.R.V.I.S. — Cognitive Core & Full-Duplex Spine",
    version="2.0.0",
    description="Hybrid Cloud-Local Cognitive Core, Full-Duplex Voice Engine, Cortex Memory Core Graph, and MCP Host Bus.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = None


class NoteUpsertRequest(BaseModel):
    title: str
    content: str
    category: str = "Knowledge"
    links: List[str] = []


class TaskCreateRequest(BaseModel):
    title: str
    priority: str = "normal"
    due_time: str = "Today"
    linked_entity: str = "[[Daily Priorities]]"


class ScheduleCreateRequest(BaseModel):
    time_slot: str
    title: str
    category: str = "Executive"


class DevZoneGenerateRequest(BaseModel):
    widget_name: str
    description: str


# ---------------------------------------------------------------------------
# Active WebSocket Connection Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(payload)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    ollama_ok = await cognitive_router.check_ollama_health()
    return {
        "status": "online",
        "system": "J.A.R.V.I.S. Cognitive Core",
        "gemini_configured": bool(cognitive_router._get_live_gemini_key()),
        "ollama_online": ollama_ok,
        "voice": voice_engine.get_status(),
        "router": cognitive_router.rate_limiter.snapshot(),
    }


@app.get("/api/telemetry")
async def api_telemetry() -> Dict[str, Any]:
    return {
        "system": get_system_telemetry(),
        "router": cognitive_router.rate_limiter.snapshot(),
        "voice": voice_engine.get_status(),
        "mcp": get_mcp_registry_state(),
    }


@app.post("/api/stt")
async def api_stt(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribes uploaded microphone audio using local Faster-Whisper."""
    raw_bytes = await audio.read()
    text = await asyncio.to_thread(voice_engine.transcribe_audio_file, raw_bytes, ".webm")
    return {"transcript": text or ""}


@app.get("/api/brain/graph")
async def api_brain_graph() -> Dict[str, Any]:
    return brain_engine.get_graph_payload()


@app.get("/api/brain/note/{node_id}")
async def api_get_note(node_id: str) -> Dict[str, Any]:
    return brain_engine.get_note(node_id)


@app.post("/api/brain/note")
async def api_upsert_note(req: NoteUpsertRequest) -> Dict[str, Any]:
    res = brain_engine.upsert_note(
        title=req.title,
        content=req.content,
        category=req.category,
        links=req.links,
    )
    graph_payload = brain_engine.get_graph_payload()
    await ws_manager.broadcast({"type": "graph_update", "graph": graph_payload})
    return {"result": res, "graph": graph_payload}


@app.delete("/api/brain/note/{node_id}")
async def api_delete_note(node_id: str) -> Dict[str, Any]:
    res = brain_engine.delete_note(node_id)
    graph_payload = brain_engine.get_graph_payload()
    await ws_manager.broadcast({"type": "graph_update", "graph": graph_payload})
    return {"result": res, "graph": graph_payload}


@app.get("/api/life/state")
async def api_life_state() -> Dict[str, Any]:
    return life_db.get_state()


@app.post("/api/life/task")
async def api_add_task(req: TaskCreateRequest) -> Dict[str, Any]:
    created = life_db.add_task(
        title=req.title,
        priority=req.priority,
        due_time=req.due_time,
        linked_entity=req.linked_entity,
    )
    state = life_db.get_state()
    await ws_manager.broadcast({"type": "life_update", "state": state})
    return {"created": created, "state": state}


@app.post("/api/life/task/{task_id}/toggle")
async def api_toggle_task(task_id: int) -> Dict[str, Any]:
    res = life_db.toggle_task(task_id)
    state = life_db.get_state()
    await ws_manager.broadcast({"type": "life_update", "state": state})
    return {"result": res, "state": state}


@app.delete("/api/life/task/{task_id}")
async def api_delete_task(task_id: int) -> Dict[str, Any]:
    res = life_db.delete_task(task_id)
    state = life_db.get_state()
    await ws_manager.broadcast({"type": "life_update", "state": state})
    return {"result": res, "state": state}


@app.post("/api/life/habit/{habit_id}/toggle")
async def api_toggle_habit(habit_id: int) -> Dict[str, Any]:
    res = life_db.toggle_habit(habit_id)
    state = life_db.get_state()
    await ws_manager.broadcast({"type": "life_update", "state": state})
    return {"result": res, "state": state}


@app.post("/api/life/schedule")
async def api_add_schedule(req: ScheduleCreateRequest) -> Dict[str, Any]:
    res = life_db.add_schedule_event(req.time_slot, req.title, req.category)
    state = life_db.get_state()
    await ws_manager.broadcast({"type": "life_update", "state": state})
    return {"result": res, "state": state}


@app.get("/api/devzone/registry")
async def api_devzone_registry() -> Dict[str, Any]:
    return get_mcp_registry_state()


@app.post("/api/devzone/generate")
async def api_devzone_generate(req: DevZoneGenerateRequest) -> Dict[str, Any]:
    exec_res = execute_mcp_tool(
        "generate_hud_widget",
        {"widget_name": req.widget_name, "description": req.description},
    )
    await ws_manager.broadcast({"type": "devzone_update", "payload": exec_res["result"]})
    return exec_res


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> Dict[str, Any]:
    if req.mode in ("auto", "cloud", "local"):
        cognitive_router.preferred_mode = req.mode
    result = await cognitive_router.route_and_respond(req.message)
    audio_b64 = await voice_engine.synthesize_speech_base64(result["spoken_reply"])
    payload = {
        **result,
        "audio_base64": audio_b64,
        "graph": brain_engine.get_graph_payload(),
        "life": life_db.get_state(),
    }
    await ws_manager.broadcast(
        {
            "type": "jarvis_response",
            "utterance": req.message,
            **payload,
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Full-Duplex Bidirectional WebSocket Spine (/ws/jarvis)
# ---------------------------------------------------------------------------
@app.websocket("/ws/jarvis")
async def jarvis_duplex_socket(websocket: WebSocket) -> None:
    """Simultaneous bidirectional audio + JSON control stream with instant barge-in cancellation."""
    await ws_manager.connect(websocket)
    active_generation_task: Optional[asyncio.Task] = None

    async def handle_utterance(text: str) -> None:
        try:
            await websocket.send_json({"type": "thinking_start", "utterance": text})
            result = await cognitive_router.route_and_respond(text)

            # Synthesize Neural British Male TTS audio (en-GB-RyanNeural)
            audio_b64 = await voice_engine.synthesize_speech_base64(result["spoken_reply"])
            voice_engine.is_jarvis_speaking = True

            await websocket.send_json(
                {
                    "type": "jarvis_response",
                    "utterance": text,
                    **result,
                    "audio_base64": audio_b64,
                    "graph": brain_engine.get_graph_payload(),
                    "life": life_db.get_state(),
                }
            )
        except asyncio.CancelledError:
            voice_engine.is_jarvis_speaking = False
            await websocket.send_json(
                {
                    "type": "generation_cancelled",
                    "reason": "user_barge_in",
                }
            )

    # Send initial handshake state
    await websocket.send_json(
        {
            "type": "handshake",
            "message": "J.A.R.V.I.S. Full-Duplex Spine Connected.",
            "graph": brain_engine.get_graph_payload(),
            "life": life_db.get_state(),
            "telemetry": {
                "system": get_system_telemetry(),
                "router": cognitive_router.rate_limiter.snapshot(),
                "voice": voice_engine.get_status(),
                "mcp": get_mcp_registry_state(),
            },
        }
    )

    try:
        while True:
            message = await websocket.receive()

            # 1. Binary 16kHz PCM Audio Frame -> Echo-Safe VAD + Barge-In check
            if "bytes" in message and message["bytes"] is not None:
                vad_res = voice_engine.process_pcm_frame(message["bytes"])
                if vad_res["barge_in_triggered"]:
                    if active_generation_task and not active_generation_task.done():
                        active_generation_task.cancel()
                    await websocket.send_json(
                        {
                            "type": "barge_in_ack",
                            "rms": vad_res["rms"],
                            "barge_in_count": voice_engine.barge_in_count,
                        }
                    )

            # 2. JSON Control / Utterance Frame
            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "barge_in":
                    ack = voice_engine.trigger_barge_in()
                    if active_generation_task and not active_generation_task.done():
                        active_generation_task.cancel()
                    await websocket.send_json({"type": "barge_in_ack", **ack})

                elif msg_type == "tts_playback_ended":
                    voice_engine.is_jarvis_speaking = False

                elif msg_type == "user_utterance":
                    text = (data.get("text") or "").strip()
                    if text:
                        if active_generation_task and not active_generation_task.done():
                            active_generation_task.cancel()
                        active_generation_task = asyncio.create_task(handle_utterance(text))

                elif msg_type == "set_router_mode":
                    mode = data.get("mode", "auto")
                    if mode in ("auto", "cloud", "local"):
                        cognitive_router.preferred_mode = mode
                    await websocket.send_json(
                        {
                            "type": "router_mode_updated",
                            "mode": cognitive_router.preferred_mode,
                            "router": cognitive_router.rate_limiter.snapshot(),
                        }
                    )

                elif msg_type == "ping_telemetry":
                    await websocket.send_json(
                        {
                            "type": "telemetry_tick",
                            "system": get_system_telemetry(),
                            "router": cognitive_router.rate_limiter.snapshot(),
                            "voice": voice_engine.get_status(),
                            "mcp": get_mcp_registry_state(),
                        }
                    )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
