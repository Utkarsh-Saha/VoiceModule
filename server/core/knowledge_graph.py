"""Part IV: The 'Cortex Memory Core' Obsidian Knowledge Graph & Hybrid Vector RAG Engine.

Combines:
1. Episodic & Semantic Vault: Local Plain-Text Markdown Vault (/brain/*.md) with YAML Frontmatter & [[WikiLinks]]
2. Relational Graph Index: SQLite + NetworkX Graph Engine parsing bidirectional [[links]], #tags, and Subject -> Predicate -> Object triples
3. Hybrid Vector RAG: Semantic retrieval with stopword filtering and 2-hop neighbor expansion (only triggered when genuine semantic relevance exists).
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from server.config import settings

WIKILINK_REGEX = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\[\]]+)?\]\]")
TAG_REGEX = re.compile(r"(?:^|\s)#([a-zA-Z0-9_-]+)")
TRIPLE_REGEX = re.compile(r"\[\[([^\]]+)\]\]\s*->\s*([a-zA-Z0-9_ -]+?)\s*->\s*\[\[([^\]]+)\]\]")
WORD_REGEX = re.compile(r"[a-zA-Z0-9_]{3,}")

STOPWORDS: Set[str] = {
    "the", "and", "for", "that", "this", "with", "you", "are", "not", "from",
    "have", "has", "had", "was", "were", "will", "would", "could", "should",
    "can", "may", "might", "what", "when", "where", "which", "who", "whom",
    "why", "how", "hey", "hello", "good", "morning", "afternoon", "evening",
    "jarvis", "sir", "checking", "working", "feature", "texting", "test",
    "testing", "just", "like", "into", "your", "about", "there", "their",
}


class CortexMemoryGraph:
    """Manages the Obsidian Markdown Vault, SQLite Relational Index, NetworkX Graph, and Hybrid RAG."""

    def __init__(self, vault_path: Optional[Path] = None, db_path: Optional[Path] = None) -> None:
        self.vault_path: Path = vault_path or settings.BRAIN_VAULT_PATH
        self.db_path: Path = db_path or settings.SQLITE_DB_PATH
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.graph: nx.Graph = nx.Graph()
        self.active_highlighted_nodes: List[str] = []
        self._session_turns: List[Dict[str, Any]] = []
        self._init_db()
        self.sync_vault()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS brain_nodes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT DEFAULT 'Note',
                    tags TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS brain_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    UNIQUE(subject, predicate, object)
                );
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    utterance TEXT NOT NULL,
                    entities TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _parse_frontmatter(raw_text: str) -> Tuple[Dict[str, str], str]:
        meta: Dict[str, str] = {}
        body = raw_text
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                body = parts[2].strip()
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip().lower()] = v.strip()
        return meta, body

    def sync_vault(self) -> Dict[str, Any]:
        """Scans /brain/*.md, indexes all [[WikiLinks]] and triples into SQLite & NetworkX."""
        self.graph.clear()
        md_files = sorted(self.vault_path.glob("*.md"))

        with self._get_conn() as conn:
            conn.execute("DELETE FROM brain_triples")
            seen_ids: Set[str] = set()

            for md_file in md_files:
                raw_text = md_file.read_text(encoding="utf-8")
                meta, body = self._parse_frontmatter(raw_text)
                title = meta.get("title", md_file.stem).strip()
                category = meta.get("category", "Note").strip()
                inline_tags = set(TAG_REGEX.findall(body))
                if "tags" in meta:
                    cleaned = meta["tags"].strip("[]")
                    for t in cleaned.split(","):
                        if t.strip():
                            inline_tags.add(t.strip())
                tags_str = ",".join(sorted(inline_tags))
                updated_at = meta.get("updated", datetime.now().strftime("%Y-%m-%d"))

                seen_ids.add(title)
                conn.execute(
                    """
                    INSERT INTO brain_nodes (id, title, category, tags, content, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        category=excluded.category,
                        tags=excluded.tags,
                        content=excluded.content,
                        updated_at=excluded.updated_at
                    """,
                    (title, title, category, tags_str, raw_text, updated_at),
                )

                self.graph.add_node(
                    title,
                    id=title,
                    label=title,
                    category=category,
                    tags=sorted(inline_tags),
                    updated_at=updated_at,
                    has_file=True,
                )

                for subj, pred, obj in TRIPLE_REGEX.findall(body):
                    s_clean, p_clean, o_clean = subj.strip(), pred.strip(), obj.strip()
                    conn.execute(
                        "INSERT OR IGNORE INTO brain_triples (subject, predicate, object) VALUES (?, ?, ?)",
                        (s_clean, p_clean, o_clean),
                    )
                    if not self.graph.has_node(o_clean):
                        self.graph.add_node(o_clean, id=o_clean, label=o_clean, category="Concept", tags=[], has_file=False)
                    self.graph.add_edge(s_clean, o_clean, relation=p_clean)

                for linked in WIKILINK_REGEX.findall(body):
                    target = linked.strip()
                    if target and target != title:
                        conn.execute(
                            "INSERT OR IGNORE INTO brain_triples (subject, predicate, object) VALUES (?, ?, ?)",
                            (title, "links_to", target),
                        )
                        if not self.graph.has_node(target):
                            self.graph.add_node(
                                target, id=target, label=target, category="Concept", tags=[], has_file=False
                            )
                        if not self.graph.has_edge(title, target):
                            self.graph.add_edge(title, target, relation="links_to")

            # Purge deleted notes from SQLite
            if seen_ids:
                placeholders = ",".join("?" for _ in seen_ids)
                conn.execute(f"DELETE FROM brain_nodes WHERE id NOT IN ({placeholders})", tuple(seen_ids))

        return self.get_graph_payload()

    def upsert_note(
        self,
        title: str,
        content: str,
        category: str = "Knowledge",
        links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Creates or updates a Markdown file in /brain/*.md with [[WikiLinks]] and re-indexes the graph."""
        safe_title = re.sub(r'[<>:"/\\|?*]', "", title).strip() or "Untitled Note"
        file_path = self.vault_path / f"{safe_title}.md"
        now_date = datetime.now().strftime("%Y-%m-%d")

        links = links or []
        existing_links = set(WIKILINK_REGEX.findall(content))
        missing_links = [lnk for lnk in links if lnk not in existing_links and lnk != safe_title]

        body = content.strip()
        if not body.startswith("---"):
            link_footer = ""
            if missing_links:
                formatted_links = " • ".join(f"[[{lnk}]]" for lnk in missing_links)
                link_footer = f"\n\n## Linked Entities\n{formatted_links}\n"
            full_markdown = (
                f"---\n"
                f"title: {safe_title}\n"
                f"category: {category}\n"
                f"tags: [{category.lower()}, jarvis-memory]\n"
                f"updated: {now_date}\n"
                f"---\n\n"
                f"# {safe_title}\n\n"
                f"{body}{link_footer}"
            )
        else:
            full_markdown = body

        file_path.write_text(full_markdown, encoding="utf-8")
        self.sync_vault()
        self.active_highlighted_nodes = [safe_title] + links[:4]
        return {
            "status": "updated",
            "node_id": safe_title,
            "file_path": str(file_path),
            "highlighted_nodes": self.active_highlighted_nodes,
        }

    def get_note(self, node_id: str) -> Dict[str, Any]:
        file_path = self.vault_path / f"{node_id}.md"
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
        else:
            content = f"---\ntitle: {node_id}\ncategory: Concept\nupdated: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n# {node_id}\n\nLinked concept node inside [[Cortex Memory Core]]."

        neighbors: List[str] = []
        if self.graph.has_node(node_id):
            neighbors = list(self.graph.neighbors(node_id))

        return {
            "id": node_id,
            "title": node_id,
            "content": content,
            "neighbors": neighbors,
        }

    def delete_note(self, title_query: str) -> Dict[str, Any]:
        """Deletes a Markdown note from /brain/*.md by exact or fuzzy title match and re-syncs the graph."""
        clean_q = title_query.strip().strip("[]").lower()
        target_file: Optional[Path] = None
        for md_file in self.vault_path.glob("*.md"):
            if md_file.stem.lower() == clean_q or clean_q in md_file.stem.lower():
                target_file = md_file
                break
        if not target_file:
            return {"status": "not_found", "query": title_query}
        deleted_title = target_file.stem
        target_file.unlink(missing_ok=True)
        self.sync_vault()
        return {"status": "deleted", "deleted_node": deleted_title}

    def log_conversation_turn(self, role: str, utterance: str) -> None:
        import time
        entities = WIKILINK_REGEX.findall(utterance)
        self._session_turns.append(
            {"role": role, "text": utterance, "ts": time.time()}
        )
        if len(self._session_turns) > 10:
            self._session_turns = self._session_turns[-10:]
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO episodic_memories (role, utterance, entities, created_at) VALUES (?, ?, ?, ?)",
                (role, utterance, ",".join(entities), datetime.now().isoformat(timespec="seconds")),
            )

    def get_recent_turns(self, limit: int = 2, max_age_seconds: float = 90.0) -> List[Dict[str, str]]:
        """Returns only fresh in-session turns (within max_age_seconds) to prevent cross-session hallucination."""
        import time
        now = time.time()
        fresh = [
            {"role": t["role"], "text": t["text"]}
            for t in self._session_turns
            if (now - t["ts"]) <= max_age_seconds
        ]
        return fresh[-limit:]

    @staticmethod
    def _tokenize(text: str) -> Counter:
        words = [w.lower() for w in WORD_REGEX.findall(text) if w.lower() not in STOPWORDS]
        return Counter(words)

    @staticmethod
    def _cosine_sim(vec_a: Counter, vec_b: Counter) -> float:
        if not vec_a or not vec_b:
            return 0.0
        intersection = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[k] * vec_b[k] for k in intersection)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve_context(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """Hybrid Vector RAG + 2-Hop NetworkX Graph Neighbor Injection.
        Only returns vault notes when there is genuine semantic/keyword relevance (>= 0.14).
        """
        q_vec = self._tokenize(query)
        if not q_vec:
            return {
                "primary_nodes": [],
                "two_hop_neighbors": [],
                "highlighted_nodes": self.active_highlighted_nodes,
                "context_block": "",
            }

        scored_nodes: List[Tuple[float, str, str]] = []
        with self._get_conn() as conn:
            rows = conn.execute("SELECT id, title, content, tags FROM brain_nodes").fetchall()

        q_low = query.lower()
        for row in rows:
            doc_text = f"{row['title']} {row['tags']} {row['content']}"
            score = self._cosine_sim(q_vec, self._tokenize(doc_text))
            if len(row["title"]) > 3 and row["title"].lower() in q_low:
                score += 0.45
            scored_nodes.append((score, row["id"], row["content"]))

        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        # Only include notes that genuinely match the query (threshold >= 0.14)
        top_matches = [item for item in scored_nodes[:top_k] if item[0] >= 0.14]

        if not top_matches:
            return {
                "primary_nodes": [],
                "two_hop_neighbors": [],
                "highlighted_nodes": self.active_highlighted_nodes,
                "context_block": "",
            }

        primary_ids = [m[1] for m in top_matches]
        two_hop_neighbors: Set[str] = set()
        for nid in primary_ids:
            if self.graph.has_node(nid):
                for hop1 in self.graph.neighbors(nid):
                    two_hop_neighbors.add(hop1)
                    for hop2 in self.graph.neighbors(hop1):
                        if hop2 != nid:
                            two_hop_neighbors.add(hop2)

        self.active_highlighted_nodes = list(dict.fromkeys(primary_ids + list(two_hop_neighbors)[:4]))

        snippets = []
        for score, nid, content in top_matches:
            _, body = self._parse_frontmatter(content)
            snippets.append(f"### [[{nid}]] (relevance={score:.2f})\n{body[:400]}")

        return {
            "primary_nodes": primary_ids,
            "two_hop_neighbors": sorted(two_hop_neighbors),
            "highlighted_nodes": self.active_highlighted_nodes,
            "context_block": "\n\n".join(snippets),
        }

    def get_graph_payload(self) -> Dict[str, Any]:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            degree = self.graph.degree(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id),
                    "category": attrs.get("category", "Note"),
                    "tags": attrs.get("tags", []),
                    "degree": degree,
                    "has_file": attrs.get("has_file", False),
                    "highlighted": node_id in self.active_highlighted_nodes,
                }
            )

        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "relation": attrs.get("relation", "links_to"),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "highlighted_nodes": self.active_highlighted_nodes,
            "total_notes": len(list(self.vault_path.glob("*.md"))),
        }


brain_engine = CortexMemoryGraph()
