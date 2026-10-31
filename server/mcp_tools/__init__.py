"""MCP Tool Registry & Dynamic Hot-Swappable Plugin Loader.

Scans `server/mcp_tools/*.py` and exposes a unified tool execution bus for J.A.R.V.I.S.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {}
TOOL_METADATA: Dict[str, Dict[str, Any]] = {}
TOOL_EXECUTION_LOG: Deque[Dict[str, Any]] = deque(maxlen=40)


def register_mcp_tool(name: str, description: str, category: str = "Core") -> Callable:
    """Decorator to register an MCP tool handler into the dynamic registry."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        TOOL_REGISTRY[name] = func
        TOOL_METADATA[name] = {
            "name": name,
            "description": description,
            "category": category,
            "module": func.__module__,
        }
        return func

    return decorator


def load_all_mcp_tools() -> Dict[str, Dict[str, Any]]:
    """Dynamically imports or reloads all modules in server/mcp_tools."""
    pkg_dir = Path(__file__).resolve().parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if is_pkg or module_name.startswith("_"):
            continue
        full_name = f"server.mcp_tools.{module_name}"
        if full_name in importlib.sys.modules:
            importlib.reload(importlib.sys.modules[full_name])
        else:
            importlib.import_module(full_name)
    return TOOL_METADATA


def execute_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a registered MCP tool and logs telemetry."""
    start = time.perf_counter()
    if name not in TOOL_REGISTRY:
        load_all_mcp_tools()

    handler = TOOL_REGISTRY.get(name)
    if not handler:
        result = {"status": "error", "error": f"Unknown MCP tool: {name}"}
    else:
        try:
            result = handler(**arguments)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}

    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    log_entry = {
        "tool": name,
        "arguments": arguments,
        "result_preview": str(result)[:160],
        "duration_ms": duration_ms,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    TOOL_EXECUTION_LOG.appendleft(log_entry)
    return {
        "tool": name,
        "arguments": arguments,
        "result": result,
        "duration_ms": duration_ms,
    }


def get_mcp_registry_state() -> Dict[str, Any]:
    load_all_mcp_tools()
    return {
        "tools": list(TOOL_METADATA.values()),
        "recent_executions": list(TOOL_EXECUTION_LOG),
    }

