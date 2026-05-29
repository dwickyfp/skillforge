"""SkillForge MCP (Model Context Protocol) server.

Implements a JSON-RPC 2.0 server over stdin/stdout that exposes SkillForge
capabilities as MCP tools.  Compatible with any MCP client such as
Claude Desktop, Hermes Agent, or custom integrations.

MCP specification reference:
    https://modelcontextprotocol.io/specification

The server implements the following MCP methods:
    - ``initialize``          – handshake, capability exchange
    - ``tools/list``          – enumerate available tools with JSON Schemas
    - ``tools/call``          – invoke a tool by name with arguments

All communication uses newline-delimited JSON-RPC 2.0 messages.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server metadata
# ---------------------------------------------------------------------------

SERVER_INFO = {
    "name": "skillforge",
    "version": "0.4.0",
}

SERVER_CAPABILITIES = {
    "tools": {"listChanged": False},
}

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for each MCP tool)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "skillforge_load_skill",
        "description": (
            "Load skills matching a natural-language query.  Returns ranked "
            "skill data at the requested detail tier."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query or keywords to match.",
                },
                "tier": {
                    "type": "integer",
                    "description": (
                        "Detail level: 1 = metadata only, 2 = core prompt, "
                        "3 = full with resources.  Default 1."
                    ),
                    "default": 1,
                    "minimum": 1,
                    "maximum": 3,
                },
                "routing": {
                    "type": "string",
                    "description": (
                        "Ranking strategy: q_value, success_rate, "
                        "usage_count, or relevance.  Default q_value."
                    ),
                    "default": "q_value",
                    "enum": ["q_value", "success_rate", "usage_count", "relevance"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of skills to return.  Default 5.",
                    "default": 5,
                    "minimum": 1,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "skillforge_record_outcome",
        "description": (
            "Record the outcome of a skill execution.  Updates the skill's "
            "Q-value, success rate, and usage counters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "The ID of the skill that produced the outcome.",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the execution was successful.",
                },
                "latency_ms": {
                    "type": ["number", "null"],
                    "description": "Wall-clock latency in milliseconds.",
                    "default": None,
                },
                "tokens_used": {
                    "type": ["integer", "null"],
                    "description": "Number of LLM tokens consumed.",
                    "default": None,
                },
            },
            "required": ["skill_id", "success"],
        },
    },
    {
        "name": "skillforge_list_skills",
        "description": (
            "List registered skills with optional filters (lifecycle, tags, "
            "limit, offset)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": ["object", "null"],
                    "description": (
                        "Optional filter dict.  Keys: lifecycle (str), "
                        "tags (list[str]), limit (int), offset (int)."
                    ),
                    "default": None,
                },
            },
        },
    },
    {
        "name": "skillforge_search_skills",
        "description": "Keyword search across skill names and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search string.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "skillforge_run_evolution",
        "description": (
            "Run a full evolution cycle across all skills.  Evaluates health, "
            "evolves under-performers, prunes dead skills, and returns a report."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "skillforge_skill_stats",
        "description": (
            "Return performance statistics for a single skill or all skills.  "
            "Pass skill_id for one skill, omit for all."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": ["string", "null"],
                    "description": (
                        "Skill ID for single-skill stats.  "
                        "Omit or pass null for all skills."
                    ),
                    "default": None,
                },
            },
        },
    },
    {
        "name": "skillforge_health_check",
        "description": (
            "Return a dashboard summary of skill health: total skills, "
            "average health, status breakdown, worst and best performers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "skillforge_diagnose",
        "description": (
            "Run self-diagnosis on a skill: analyze recent failures and "
            "return actionable insights with patch suggestions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "The skill to diagnose.",
                },
            },
            "required": ["skill_id"],
        },
    },
]

# Build a quick lookup map.
_TOOL_MAP: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOL_DEFINITIONS}


# ---------------------------------------------------------------------------
# Helper: serialise Skill objects
# ---------------------------------------------------------------------------

def _skill_to_dict(skill: Any) -> dict[str, Any]:
    """Convert a Skill dataclass to a JSON-safe dict."""
    d: dict[str, Any] = {}
    for key in ("id", "name", "version", "tier1_metadata", "tier2_core",
                "tier3_resources", "q_value", "success_rate", "usage_count",
                "tags"):
        val = getattr(skill, key, None)
        d[key] = val
    lifecycle = getattr(skill, "lifecycle", None)
    d["lifecycle"] = lifecycle.value if lifecycle is not None else None
    created = getattr(skill, "created_at", None)
    d["created_at"] = created.isoformat() if created is not None else None
    updated = getattr(skill, "updated_at", None)
    d["updated_at"] = updated.isoformat() if updated is not None else None
    return d


# ---------------------------------------------------------------------------
# SkillForgeMCPServer
# ---------------------------------------------------------------------------

class SkillForgeMCPServer:
    """MCP server that exposes SkillForge tools over JSON-RPC on stdin/stdout.

    Parameters
    ----------
    skillforge_or_db_path : SkillForge | str | Path
        Either an existing :class:`~skillforge.forge.SkillForge` instance or a
        path string to the SQLite database.  When a string/Path is given the
        server creates a SkillForge instance lazily on first use.
    """

    def __init__(
        self,
        skillforge_or_db_path: Any = "~/.skillforge/skillforge.db",
    ) -> None:
        self._forge_arg = skillforge_or_db_path
        self._forge: Any = None  # lazy

    # -- lazy forge access --------------------------------------------------

    @property
    def forge(self) -> Any:
        """Return (and lazily create) the SkillForge instance."""
        if self._forge is None:
            from skillforge.forge import SkillForge

            if isinstance(self._forge_arg, str):
                self._forge = SkillForge(db_path=self._forge_arg)
            elif isinstance(self._forge_arg, Path):
                self._forge = SkillForge(db_path=str(self._forge_arg))
            else:
                # Assume it's already a SkillForge instance.
                self._forge = self._forge_arg
        return self._forge

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Read JSON-RPC messages from stdin, write responses to stdout.

        Messages are newline-delimited.  Each line is a complete JSON-RPC 2.0
        request or notification.  The server blocks on stdin until EOF.
        """
        logger.info("SkillForge MCP server starting (stdin/stdout)")
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                error_response = self._error_response(
                    None, -32700, f"Parse error: {exc}"
                )
                self._send(error_response)
                continue

            response = self.handle_request(request)
            if response is not None:
                self._send(response)

    # -- request routing ----------------------------------------------------

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Route a JSON-RPC request to the appropriate handler.

        Parameters
        ----------
        request : dict
            A JSON-RPC 2.0 request object.

        Returns
        -------
        dict | None
            A JSON-RPC response dict, or *None* for notifications (no id).
        """
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # Notifications (no id) are silently ignored.
        if req_id is None:
            return None

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(
                    req_id, -32601, f"Method not found: {method}"
                )

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        except Exception as exc:
            logger.exception("Error handling %s", method)
            return self._error_response(req_id, -32603, str(exc))

    # -- MCP method handlers ------------------------------------------------

    @staticmethod
    def _handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
        """Return server info and capabilities."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": SERVER_INFO,
            "capabilities": SERVER_CAPABILITIES,
        }

    @staticmethod
    def _handle_tools_list() -> dict[str, Any]:
        """Return the list of available tool definitions."""
        return {"tools": TOOL_DEFINITIONS}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate handler."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in _TOOL_MAP:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            raise ValueError(f"No handler for tool: {tool_name}")

        result = handler(**arguments)
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, default=str)},
            ],
        }

    # -- individual tool implementations -----------------------------------

    def _tool_skillforge_load_skill(
        self,
        query: str,
        tier: int = 1,
        routing: str = "q_value",
        limit: int = 5,
        **_: Any,
    ) -> list[dict[str, Any]]:
        skills = self.forge.load_skill(
            query, tier=tier, routing=routing, limit=limit
        )
        return [_skill_to_dict(s) for s in skills]

    def _tool_skillforge_record_outcome(
        self,
        skill_id: str,
        success: bool,
        latency_ms: float | None = None,
        tokens_used: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.forge.record_outcome(
            skill_id,
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        return {
            "status": "ok",
            "skill_id": skill_id,
            "success": success,
        }

    def _tool_skillforge_list_skills(
        self,
        filters: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        skills = self.forge.list_skills(filters=filters)
        return [_skill_to_dict(s) for s in skills]

    def _tool_skillforge_search_skills(
        self,
        query: str,
        **_: Any,
    ) -> list[dict[str, Any]]:
        skills = self.forge.search_skills(query)
        return [_skill_to_dict(s) for s in skills]

    def _tool_skillforge_run_evolution(self, **_: Any) -> dict[str, Any]:
        report = self.forge.run_evolution_loop()
        return report.to_dict()

    def _tool_skillforge_skill_stats(
        self,
        skill_id: str | None = None,
        **_: Any,
    ) -> Any:
        return self.forge.get_skill_stats(skill_id=skill_id)

    def _tool_skillforge_health_check(self, **_: Any) -> dict[str, Any]:
        """Return health dashboard summary.

        Uses the HealthMonitor intelligence layer if available, otherwise
        falls back to basic stats from the forge.
        """
        try:
            from skillforge.intelligence.health_monitor import HealthMonitor

            monitor = HealthMonitor(
                registry=self.forge._registry,
                tracker=self.forge._tracker,
                graph=self.forge._graph,
            )
            return monitor.get_dashboard_summary()
        except Exception:
            # Fallback: build a basic summary from skill stats.
            all_stats = self.forge.get_skill_stats()
            if not isinstance(all_stats, list) or not all_stats:
                return {
                    "total_skills": 0,
                    "avg_health": 0.0,
                    "by_status": {"healthy": 0, "warning": 0, "critical": 0},
                }
            total = len(all_stats)
            avg_q = sum(s.get("q_value", 0.5) for s in all_stats) / total
            healthy = sum(1 for s in all_stats if s.get("q_value", 0.5) >= 0.7)
            warning = sum(1 for s in all_stats if 0.4 <= s.get("q_value", 0.5) < 0.7)
            critical = sum(1 for s in all_stats if s.get("q_value", 0.5) < 0.4)
            return {
                "total_skills": total,
                "avg_health": round(avg_q, 4),
                "by_status": {
                    "healthy": healthy,
                    "warning": warning,
                    "critical": critical,
                },
            }

    def _tool_skillforge_diagnose(
        self,
        skill_id: str,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Run self-diagnosis on a skill, with graceful fallback."""
        try:
            insights = self.forge._diagnosis.analyze_failures(skill_id)
            return [
                i.to_dict() if hasattr(i, "to_dict") else {"text": str(i)}
                for i in insights
            ]
        except AttributeError:
            # The tracker may not implement the full diagnosis protocol
            # (e.g. missing get_failures).  Fall back to basic stats.
            try:
                stats = self.forge.get_skill_stats(skill_id=skill_id)
                return [{
                    "root_cause": "Basic diagnosis (full failure analysis unavailable)",
                    "heuristic": "stats_fallback",
                    "patch_suggestion": (
                        f"Q-value: {stats.get('q_value', 'N/A')}, "
                        f"Success rate: {stats.get('success_rate', 'N/A')}, "
                        f"Total outcomes: {stats.get('total_outcomes', 'N/A')}"
                    ),
                    "confidence": 0.3,
                }]
            except Exception as exc:
                return [{
                    "root_cause": f"Unable to diagnose: {exc}",
                    "heuristic": "error",
                    "patch_suggestion": "Check skill_id and ensure skill is registered.",
                    "confidence": 0.0,
                }]

    # -- JSON-RPC helpers ---------------------------------------------------

    @staticmethod
    def _error_response(
        req_id: Any, code: int, message: str
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _send(response: dict[str, Any]) -> None:
        """Write a JSON-RPC response followed by a newline to stdout."""
        line = json.dumps(response, default=str)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Create and run the MCP server.

    Intended for use as ``python -m skillforge.mcp.server`` or as an MCP
    transport command in client configuration.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    server = SkillForgeMCPServer()
    server.run()


if __name__ == "__main__":
    main()
