"""MCP Tool Module: 'Cortex Memory Core' Obsidian Knowledge Graph Operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from server.core.knowledge_graph import brain_engine
from server.mcp_tools import register_mcp_tool


@register_mcp_tool(
    name="update_knowledge_graph",
    description="Create or update a Markdown note in /brain/*.md with [[WikiLinks]] and re-index the force graph.",
    category="Cortex Memory Core",
)
def update_knowledge_graph(
    title: str,
    content: str,
    category: str = "Knowledge",
    links: Optional[List[str]] = None,
) -> Dict[str, Any]:
    res = brain_engine.upsert_note(
        title=title,
        content=content,
        category=category,
        links=links or ["Sir", "Project J.A.R.V.I.S."],
    )
    return {
        "hud_action": "graph_updated",
        "node_id": res["node_id"],
        "highlighted_nodes": res["highlighted_nodes"],
        "graph": brain_engine.get_graph_payload(),
    }


@register_mcp_tool(
    name="delete_knowledge_node",
    description="Delete a Markdown note from /brain/*.md by title and re-index the Cortex Memory Graph.",
    category="Cortex Memory Core",
)
def delete_knowledge_node(title: str) -> Dict[str, Any]:
    res = brain_engine.delete_note(title)
    return {
        "hud_action": "graph_updated",
        "deleted": res,
        "graph": brain_engine.get_graph_payload(),
    }


@register_mcp_tool(
    name="query_knowledge_graph",
    description="Retrieve top-K semantic chunks and 2-hop NetworkX neighbors from the Cortex Memory Vault.",
    category="Cortex Memory Core",
)
def query_knowledge_graph(query: str, top_k: int = 3) -> Dict[str, Any]:
    return brain_engine.retrieve_context(query=query, top_k=top_k)
