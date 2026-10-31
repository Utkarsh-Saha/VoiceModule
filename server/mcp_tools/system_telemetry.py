"""MCP Tool Module: System Telemetry (CPU, RAM, Disk, Battery, Network)."""

from __future__ import annotations

import platform
from typing import Any, Dict

import psutil

from server.mcp_tools import register_mcp_tool


@register_mcp_tool(
    name="get_system_telemetry",
    description="Collect real-time host CPU, RAM, Disk, Battery, and Network telemetry.",
    category="System Telemetry",
)
def get_system_telemetry() -> Dict[str, Any]:
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    battery = psutil.sensors_battery()
    net = psutil.net_io_counters()

    return {
        "cpu_percent": round(cpu_pct, 1),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_percent": round(vm.percent, 1),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "disk_percent": round(disk.percent, 1),
        "battery_percent": round(battery.percent, 1) if battery else 100.0,
        "power_plugged": bool(battery.power_plugged) if battery else True,
        "net_sent_mb": round(net.bytes_sent / (1024**2), 1),
        "net_recv_mb": round(net.bytes_recv / (1024**2), 1),
        "platform": f"{platform.system()} {platform.release()}",
    }

