"""SkillForge HTTP API server built on ``http.server``.

Provides a threaded REST API that exposes every SkillForge operation as a
JSON-over-HTTP endpoint.  The server uses **only** the Python standard
library — no third-party web frameworks required.

Typical usage::

    from skillforge import SkillForge
    from skillforge.api import SkillForgeAPIServer

    forge = SkillForge()
    server = SkillForgeAPIServer(forge)
    server.start()
    # … server is now listening on http://127.0.0.1:8742
    server.stop()
    forge.close()
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skillforge.forge import SkillForge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route table — compiled regex patterns for path matching
# ---------------------------------------------------------------------------

_ROUTE_SKILLS_COLLECTION = re.compile(r"^/api/v1/skills/?$")
_ROUTE_SKILL_DETAIL = re.compile(r"^/api/v1/skills/([^/]+)/?$")
_ROUTE_SKILL_STATS = re.compile(r"^/api/v1/skills/([^/]+)/stats/?$")
_ROUTE_SKILLS_LOAD = re.compile(r"^/api/v1/skills/load/?$")
_ROUTE_OUTCOMES = re.compile(r"^/api/v1/outcomes/?$")
_ROUTE_EVOLUTION = re.compile(r"^/api/v1/evolution/?$")
_ROUTE_HEALTH = re.compile(r"^/api/v1/health/?$")
_ROUTE_DASHBOARD = re.compile(r"^/api/v1/dashboard/?$")
_ROUTE_ANALYTICS = re.compile(r"^/api/v1/analytics/?$")
_ROUTE_CONFLICTS = re.compile(r"^/api/v1/conflicts/?$")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    """Read and parse the JSON body from an incoming HTTP request.

    Parameters
    ----------
    handler : BaseHTTPRequestHandler
        The active request handler.

    Returns
    -------
    dict
        Parsed JSON body.  Returns an empty dict when the body is absent or
        empty.

    Raises
    ------
    ValueError
        If the body is not valid JSON.
    """
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        return {}
    raw = handler.rfile.read(content_length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def json_response(
    handler: BaseHTTPRequestHandler,
    data: Any,
    status: int = 200,
) -> None:
    """Send a JSON response with appropriate headers.

    Parameters
    ----------
    handler : BaseHTTPRequestHandler
        The active request handler.
    data : Any
        Data to serialise as JSON.
    status : int
        HTTP status code.
    """
    body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.end_headers()
    handler.wfile.write(body)


def parse_path_params(
    pattern: re.Pattern[str],
    path: str,
) -> tuple[str, ...] | None:
    """Match *path* against a compiled regex and return captured groups.

    Parameters
    ----------
    pattern : re.Pattern
        Compiled regex with at least one capturing group.
    path : str
        The request path to match.

    Returns
    -------
    tuple[str, ...] | None
        Captured groups if the pattern matches, otherwise *None*.
    """
    match = pattern.match(path)
    if match is None:
        return None
    return match.groups()


# ---------------------------------------------------------------------------
# Skill serialisation helper
# ---------------------------------------------------------------------------


def _skill_to_dict(skill: Any, tier: int = 1) -> dict[str, Any]:
    """Convert a Skill dataclass to a JSON-friendly dict.

    Parameters
    ----------
    skill : Skill
        The skill instance.
    tier : int
        Detail level to include (1 = metadata, 2 = +core, 3 = +resources).

    Returns
    -------
    dict
        Serialised skill.
    """
    d: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "version": skill.version,
        "lifecycle": skill.lifecycle.value if hasattr(skill.lifecycle, "value") else str(skill.lifecycle),
        "tier1_metadata": skill.tier1_metadata,
        "q_value": skill.q_value,
        "success_rate": skill.success_rate,
        "usage_count": skill.usage_count,
        "tags": skill.tags,
        "created_at": skill.created_at.isoformat() if isinstance(skill.created_at, datetime) else str(skill.created_at),
        "updated_at": skill.updated_at.isoformat() if isinstance(skill.updated_at, datetime) else str(skill.updated_at),
    }
    if tier >= 2:
        d["tier2_core"] = skill.tier2_core
    if tier >= 3:
        d["tier3_resources"] = skill.tier3_resources
    return d


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the SkillForge REST API.

    Routes all requests through a central ``_dispatch`` method that matches
    the path against compiled regex patterns and delegates to specific
    handler methods.

    Attributes
    ----------
    server : SkillForgeHTTPServer
        The parent HTTP server instance (provides access to the forge).
    """

    # Allow narrowed type for the forge-aware server
    server: "SkillForgeHTTPServer"  # type: ignore[assignment]

    # Silence default stderr logging from BaseHTTPRequestHandler
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Redirect HTTP log lines to the module logger."""
        logger.debug("HTTP %s %s", self.command, self.path)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        """Dispatch GET requests."""
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        """Dispatch POST requests."""
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        """Dispatch PUT requests."""
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        """Dispatch DELETE requests."""
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        """Route the request to the appropriate handler.

        Parameters
        ----------
        method : str
            HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``).
        """
        path = self.path.split("?")[0]  # strip query string

        try:
            # Health check
            if method == "GET" and _ROUTE_HEALTH.match(path):
                self._handle_health()
            # Dashboard
            elif method == "GET" and _ROUTE_DASHBOARD.match(path):
                self._handle_dashboard()
            # Analytics
            elif method == "GET" and _ROUTE_ANALYTICS.match(path):
                self._handle_analytics()
            # Conflicts
            elif method == "GET" and _ROUTE_CONFLICTS.match(path):
                self._handle_conflicts()
            # Skills load (POST)
            elif method == "POST" and _ROUTE_SKILLS_LOAD.match(path):
                self._handle_skills_load()
            # Skills collection (GET/POST)
            elif method == "GET" and _ROUTE_SKILLS_COLLECTION.match(path):
                self._handle_list_skills()
            elif method == "POST" and _ROUTE_SKILLS_COLLECTION.match(path):
                self._handle_create_skill()
            # Skill stats (GET) — must be checked before detail route
            elif method == "GET" and (params := parse_path_params(_ROUTE_SKILL_STATS, path)):
                self._handle_skill_stats(params[0])
            # Skill detail (GET/PUT/DELETE)
            elif method == "GET" and (params := parse_path_params(_ROUTE_SKILL_DETAIL, path)):
                self._handle_get_skill(params[0])
            elif method == "PUT" and (params := parse_path_params(_ROUTE_SKILL_DETAIL, path)):
                self._handle_update_skill(params[0])
            elif method == "DELETE" and (params := parse_path_params(_ROUTE_SKILL_DETAIL, path)):
                self._handle_deprecate_skill(params[0])
            # Outcomes
            elif method == "POST" and _ROUTE_OUTCOMES.match(path):
                self._handle_record_outcome()
            # Evolution
            elif method == "POST" and _ROUTE_EVOLUTION.match(path):
                self._handle_evolution()
            # 404 fallback
            else:
                json_response(self, {"error": "Not found", "path": path}, status=404)
        except ValueError as exc:
            json_response(self, {"error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("Unhandled error in request handler")
            json_response(
                self,
                {"error": "Internal server error", "detail": str(exc)},
                status=500,
            )

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        """GET /api/v1/health — health check."""
        forge = self.server.forge
        skills = forge.list_skills()
        json_response(self, {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_skills": len(skills),
            "server": "skillforge-api",
            "version": "0.1.0",
        })

    def _handle_dashboard(self) -> None:
        """GET /api/v1/dashboard — dashboard summary."""
        forge = self.server.forge
        all_stats = forge.get_skill_stats()
        if not isinstance(all_stats, list):
            all_stats = []

        total_skills = len(all_stats)
        healthy = sum(1 for s in all_stats if s.get("q_value", 0) >= 0.6)
        warning = sum(1 for s in all_stats if 0.3 <= s.get("q_value", 0) < 0.6)
        critical = sum(1 for s in all_stats if s.get("q_value", 0) < 0.3)

        avg_q = (
            sum(s.get("q_value", 0.5) for s in all_stats) / total_skills
            if total_skills
            else 0.0
        )
        total_outcomes = sum(s.get("total_outcomes", 0) for s in all_stats)

        json_response(self, {
            "total_skills": total_skills,
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "average_q_value": round(avg_q, 4),
            "total_outcomes": total_outcomes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_analytics(self) -> None:
        """GET /api/v1/analytics — analytics report."""
        forge = self.server.forge
        all_stats = forge.get_skill_stats()
        if not isinstance(all_stats, list):
            all_stats = []

        # Top skills by Q-value
        sorted_skills = sorted(
            all_stats, key=lambda s: s.get("q_value", 0), reverse=True
        )
        top_skills = [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "q_value": s.get("q_value", 0),
                "success_rate": s.get("success_rate", 0),
                "total_outcomes": s.get("total_outcomes", 0),
            }
            for s in sorted_skills[:10]
        ]

        # Aggregate metrics
        total_outcomes = sum(s.get("total_outcomes", 0) for s in all_stats)
        avg_latency = (
            sum(s.get("avg_latency_ms", 0) for s in all_stats) / len(all_stats)
            if all_stats
            else 0.0
        )
        avg_success = (
            sum(s.get("success_rate", 0.5) for s in all_stats) / len(all_stats)
            if all_stats
            else 0.0
        )

        json_response(self, {
            "total_skills": len(all_stats),
            "total_outcomes": total_outcomes,
            "average_latency_ms": round(avg_latency, 2),
            "average_success_rate": round(avg_success, 4),
            "top_skills": top_skills,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_conflicts(self) -> None:
        """GET /api/v1/conflicts — list potential skill conflicts."""
        forge = self.server.forge
        skills = forge.list_skills()

        # Detect potential conflicts: skills with overlapping tags
        # and similar names (simple heuristic)
        conflicts: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, s1 in enumerate(skills):
            for s2 in skills[i + 1:]:
                pair: tuple[str, str] = (min(s1.id, s2.id), max(s1.id, s2.id))
                if pair in seen_pairs:
                    continue
                # Check for tag overlap
                shared_tags = set(s1.tags) & set(s2.tags)
                if shared_tags:
                    seen_pairs.add(pair)
                    conflicts.append({
                        "skill_a": {"id": s1.id, "name": s1.name},
                        "skill_b": {"id": s2.id, "name": s2.name},
                        "shared_tags": sorted(shared_tags),
                        "severity": "warning" if len(shared_tags) > 2 else "info",
                    })

        json_response(self, {
            "conflicts": conflicts,
            "total": len(conflicts),
        })

    def _handle_list_skills(self) -> None:
        """GET /api/v1/skills — list all skills."""
        forge = self.server.forge
        skills = forge.list_skills()
        json_response(self, {
            "skills": [_skill_to_dict(s) for s in skills],
            "total": len(skills),
        })

    def _handle_get_skill(self, skill_id: str) -> None:
        """GET /api/v1/skills/{id} — get skill detail."""
        forge = self.server.forge
        # Try full detail (tier 3)
        skill = forge._registry.get_skill(skill_id, tier=3)
        if skill is None:
            json_response(self, {"error": f"Skill '{skill_id}' not found"}, status=404)
            return
        json_response(self, {"skill": _skill_to_dict(skill, tier=3)})

    def _handle_create_skill(self) -> None:
        """POST /api/v1/skills — register a new skill."""
        body = parse_json_body(self)
        name = body.get("name")
        if not name:
            raise ValueError("'name' is required in request body")

        forge = self.server.forge
        skill = forge.register_skill(
            name=name,
            tier1_metadata=body.get("tier1_metadata", body.get("description", "")),
            tier2_core=body.get("tier2_core", body.get("instructions", "")),
            tier3_resources=body.get("tier3_resources", body.get("resources", [])),
            tags=body.get("tags", []),
            skill_id=body.get("skill_id", body.get("id")),
        )
        json_response(self, {"skill": _skill_to_dict(skill, tier=3)}, status=201)

    def _handle_update_skill(self, skill_id: str) -> None:
        """PUT /api/v1/skills/{id} — update skill."""
        body = parse_json_body(self)
        if not body:
            raise ValueError("Request body must contain fields to update")

        forge = self.server.forge
        skill = forge._registry.update_skill(skill_id, body)
        if skill is None:
            json_response(self, {"error": f"Skill '{skill_id}' not found"}, status=404)
            return
        json_response(self, {"skill": _skill_to_dict(skill, tier=3)})

    def _handle_deprecate_skill(self, skill_id: str) -> None:
        """DELETE /api/v1/skills/{id} — deprecate skill."""
        forge = self.server.forge
        skill = forge._registry.update_skill(
            skill_id, {"lifecycle": "deprecated"}
        )
        if skill is None:
            json_response(self, {"error": f"Skill '{skill_id}' not found"}, status=404)
            return
        json_response(self, {
            "message": f"Skill '{skill_id}' deprecated",
            "skill": _skill_to_dict(skill),
        })

    def _handle_skill_stats(self, skill_id: str) -> None:
        """GET /api/v1/skills/{id}/stats — get skill statistics."""
        forge = self.server.forge
        # Verify skill exists
        skill = forge._registry.get_skill(skill_id, tier=1)
        if skill is None:
            json_response(self, {"error": f"Skill '{skill_id}' not found"}, status=404)
            return
        stats = forge.get_skill_stats(skill_id)
        json_response(self, {"stats": stats})

    def _handle_skills_load(self) -> None:
        """POST /api/v1/skills/load — progressive load."""
        body = parse_json_body(self)
        query = body.get("query")
        if not query:
            raise ValueError("'query' is required in request body")

        tier = int(body.get("tier", 1))
        routing = body.get("routing", "q_value")
        limit = int(body.get("limit", 5))

        forge = self.server.forge
        skills = forge.load_skill(query, tier=tier, routing=routing, limit=limit)
        json_response(self, {
            "skills": [_skill_to_dict(s, tier=tier) for s in skills],
            "total": len(skills),
            "query": query,
            "tier": tier,
            "routing": routing,
        })

    def _handle_record_outcome(self) -> None:
        """POST /api/v1/outcomes — record an execution outcome."""
        body = parse_json_body(self)
        skill_id = body.get("skill_id")
        if not skill_id:
            raise ValueError("'skill_id' is required in request body")
        if "success" not in body:
            raise ValueError("'success' is required in request body")

        forge = self.server.forge
        forge.record_outcome(
            skill_id=skill_id,
            success=bool(body["success"]),
            latency_ms=body.get("latency_ms"),
            tokens_used=body.get("tokens_used"),
            user_feedback=body.get("user_feedback"),
        )
        json_response(self, {
            "message": "Outcome recorded",
            "skill_id": skill_id,
            "success": bool(body["success"]),
        }, status=201)

    def _handle_evolution(self) -> None:
        """POST /api/v1/evolution — trigger evolution loop."""
        body = parse_json_body(self)
        thresholds = body.get("thresholds")

        forge = self.server.forge
        report = forge.run_evolution_loop(thresholds=thresholds)
        json_response(self, {
            "message": "Evolution cycle complete",
            "report": report.to_dict(),
        })


# ---------------------------------------------------------------------------
# HTTP server wrapper
# ---------------------------------------------------------------------------


class SkillForgeHTTPServer(HTTPServer):
    """HTTPServer subclass that carries a reference to the SkillForge instance.

    Parameters
    ----------
    forge : SkillForge
        The forge instance to expose via the API.
    server_address : tuple
        ``(host, port)`` binding tuple.
    """

    def __init__(
        self,
        forge: "SkillForge",
        server_address: tuple[str, int],
    ) -> None:
        self.forge = forge
        super().__init__(server_address, RequestHandler)
        # Allow SQLite connections to be used across threads (the HTTP
        # server handles requests on a separate thread from the one that
        # created the forge and its connections).
        self._enable_cross_thread_sqlite()

    def _enable_cross_thread_sqlite(self) -> None:
        """Recreate SQLite connections with ``check_same_thread=False``.

        The HTTP server handles requests on a separate thread from the one
        that created the forge and its connections.  SQLite rejects
        cross-thread usage by default, so we reconnect with the flag
        disabled.
        """
        import sqlite3

        for attr_name in ("_registry", "_tracker"):
            component = getattr(self.forge, attr_name, None)
            if component is None:
                continue
            conn = getattr(component, "_conn", None)
            if conn is None:
                continue

            db_path = getattr(component, "_db_path", None)
            if db_path is None:
                continue

            # Close the old connection and open a new one that allows
            # cross-thread access.
            try:
                conn.close()
            except Exception:
                pass
            new_conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
            )
            new_conn.row_factory = sqlite3.Row
            new_conn.execute("PRAGMA journal_mode=WAL")
            new_conn.execute("PRAGMA foreign_keys=ON")
            component._conn = new_conn


# ---------------------------------------------------------------------------
# Public API server class
# ---------------------------------------------------------------------------


class SkillForgeAPIServer:
    """High-level wrapper around :class:`SkillForgeHTTPServer`.

    Manages the HTTP server lifecycle and runs it in a background daemon
    thread so that the calling process can continue doing other work.

    Parameters
    ----------
    skillforge : SkillForge
        The forge instance to serve.
    host : str
        Bind address.  Defaults to ``127.0.0.1`` (loopback only).
    port : int
        TCP port.  Defaults to ``8742``.

    Examples
    --------
    >>> server = SkillForgeAPIServer(forge, port=9000)
    >>> server.start()
    >>> server.is_running()
    True
    >>> server.stop()
    """

    def __init__(
        self,
        skillforge: "SkillForge | None" = None,
        host: str = "127.0.0.1",
        port: int = 8742,
        *,
        db_path: str | None = None,
    ) -> None:
        if skillforge is None and db_path is not None:
            from skillforge.forge import SkillForge as _SkillForge
            skillforge = _SkillForge(db_path=db_path)
        if skillforge is None:
            raise ValueError("Either skillforge or db_path must be provided")
        self._forge = skillforge
        self._host = host
        self._port = port
        self._server: SkillForgeHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread.

        The thread is a daemon so it will not prevent the interpreter from
        shutting down when the main thread exits.

        Raises
        ------
        RuntimeError
            If the server is already running.
        """
        if self._server is not None:
            raise RuntimeError(
                f"Server is already running on {self._host}:{self._port}"
            )

        self._server = SkillForgeHTTPServer(
            forge=self._forge,
            server_address=(self._host, self._port),
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="skillforge-api",
            daemon=True,
        )
        self._thread.start()
        logger.info("SkillForge API server started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Gracefully stop the HTTP server and join its thread.

        Safe to call multiple times or when the server is not running.
        """
        if self._server is None:
            return
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server.server_close()
        logger.info("SkillForge API server stopped")
        self._server = None
        self._thread = None

    def is_running(self) -> bool:
        """Return ``True`` if the server is currently running.

        Returns
        -------
        bool
            Whether the server thread is alive.
        """
        return self._thread is not None and self._thread.is_alive()

    @property
    def host(self) -> str:
        """The bind address."""
        return self._host

    @property
    def port(self) -> int:
        """The TCP port."""
        return self._port

    @property
    def url(self) -> str:
        """The base URL of the running server."""
        return f"http://{self._host}:{self._port}"
