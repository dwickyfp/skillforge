"""Python client for the SkillForge REST API.

Communicates with a running :class:`~skillforge.api.server.SkillForgeAPIServer`
over HTTP using only the standard library (``urllib.request``).  No third-party
dependencies such as ``requests`` are required.

Typical usage::

    from skillforge.api import SkillForgeClient

    client = SkillForgeClient("http://127.0.0.1:8742")
    health = client.get_health()
    skills = client.list_skills()
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SkillForgeClientError(Exception):
    """Raised when the API returns an error response.

    Attributes
    ----------
    status_code : int
        HTTP status code.
    detail : str
        Error message from the server.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class SkillForgeClient:
    """Lightweight HTTP client for the SkillForge REST API.

    All methods return parsed JSON (``dict`` or ``list``) and raise
    :class:`SkillForgeClientError` on non-2xx responses.

    Parameters
    ----------
    base_url : str
        Root URL of the API server, e.g. ``"http://127.0.0.1:8742"``.
    timeout : float
        Socket timeout in seconds for every request.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8742",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Send an HTTP request and return the parsed JSON response.

        Parameters
        ----------
        method : str
            HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``).
        path : str
            API path (appended to the base URL).
        body : dict | None
            Optional JSON body for POST/PUT.

        Returns
        -------
        Any
            Parsed JSON response.

        Raises
        ------
        SkillForgeClientError
            On non-2xx HTTP responses or connection errors.
        """
        url = f"{self._base_url}{path}"
        data: bytes | None = None
        headers: dict[str, str] = {
            "Accept": "application/json",
        }

        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url, data=data, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
                detail = error_body.get("error", str(exc))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = str(exc)
            raise SkillForgeClientError(exc.code, detail) from exc
        except URLError as exc:
            raise SkillForgeClientError(
                0, f"Connection error: {exc.reason}"
            ) from exc

    def _get(self, path: str) -> Any:
        """Send a GET request."""
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """Send a POST request."""
        return self._request("POST", path, body=body or {})

    def _put(self, path: str, body: dict[str, Any]) -> Any:
        """Send a PUT request."""
        return self._request("PUT", path, body=body)

    def _delete(self, path: str) -> Any:
        """Send a DELETE request."""
        return self._request("DELETE", path)

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills.

        Returns
        -------
        list[dict]
            List of skill dictionaries.
        """
        result = self._get("/api/v1/skills")
        return result.get("skills", [])

    def get_skill(self, skill_id: str) -> dict[str, Any]:
        """Get full detail for a single skill.

        Parameters
        ----------
        skill_id : str
            The skill's unique identifier.

        Returns
        -------
        dict
            Skill detail including all tiers.
        """
        result = self._get(f"/api/v1/skills/{skill_id}")
        return result.get("skill", result)

    def create_skill(
        self,
        name: str,
        description: str = "",
        instructions: str = "",
        resources: list[str] | None = None,
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a new skill.

        Parameters
        ----------
        name : str
            Human-readable skill name.
        description : str
            Short summary (~30 tokens) for routing.
        instructions : str
            Full core prompt / instructions.
        resources : list[str] | None
            Auxiliary file paths or URLs.
        tags : list[str] | None
            Descriptive tags.
        skill_id : str | None
            Explicit ID (auto-generated if omitted).

        Returns
        -------
        dict
            The newly created skill.
        """
        body: dict[str, Any] = {
            "name": name,
            "tier1_metadata": description,
            "tier2_core": instructions,
        }
        if resources is not None:
            body["tier3_resources"] = resources
        if tags is not None:
            body["tags"] = tags
        if skill_id is not None:
            body["skill_id"] = skill_id

        result = self._post("/api/v1/skills", body)
        return result.get("skill", result)

    def update_skill(
        self,
        skill_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update fields on an existing skill.

        Parameters
        ----------
        skill_id : str
            The skill's unique identifier.
        updates : dict
            Fields to update (e.g. ``{"name": "new name"}``).

        Returns
        -------
        dict
            The updated skill.
        """
        result = self._put(f"/api/v1/skills/{skill_id}", updates)
        return result.get("skill", result)

    def deprecate_skill(self, skill_id: str) -> dict[str, Any]:
        """Deprecate a skill (soft-delete via lifecycle change).

        Parameters
        ----------
        skill_id : str
            The skill's unique identifier.

        Returns
        -------
        dict
            Confirmation with the updated skill.
        """
        return self._delete(f"/api/v1/skills/{skill_id}")

    def load_skill(
        self,
        query: str,
        tier: int = 1,
        routing: str = "q_value",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Progressively load skills matching a query.

        Parameters
        ----------
        query : str
            Natural-language search query.
        tier : int
            Maximum detail level (1–3).
        routing : str
            Ranking strategy (``q_value``, ``success_rate``, etc.).
        limit : int
            Maximum results to return.

        Returns
        -------
        dict
            Response with ``skills`` list and metadata.
        """
        return self._post("/api/v1/skills/load", {
            "query": query,
            "tier": tier,
            "routing": routing,
            "limit": limit,
        })

    def get_skill_stats(self, skill_id: str) -> dict[str, Any]:
        """Get performance statistics for a single skill.

        Parameters
        ----------
        skill_id : str
            The skill's unique identifier.

        Returns
        -------
        dict
            Statistics dictionary.
        """
        result = self._get(f"/api/v1/skills/{skill_id}/stats")
        return result.get("stats", result)

    def record_outcome(
        self,
        skill_id: str,
        success: bool,
        latency_ms: float | None = None,
        tokens_used: int | None = None,
        user_feedback: float | None = None,
    ) -> dict[str, Any]:
        """Record an execution outcome for a skill.

        Parameters
        ----------
        skill_id : str
            The skill that produced the outcome.
        success : bool
            Whether the execution was successful.
        latency_ms : float | None
            Wall-clock latency in milliseconds.
        tokens_used : int | None
            Number of LLM tokens consumed.
        user_feedback : float | None
            Optional user feedback score (0–5).

        Returns
        -------
        dict
            Confirmation response.
        """
        body: dict[str, Any] = {
            "skill_id": skill_id,
            "success": success,
        }
        if latency_ms is not None:
            body["latency_ms"] = latency_ms
        if tokens_used is not None:
            body["tokens_used"] = tokens_used
        if user_feedback is not None:
            body["user_feedback"] = user_feedback

        return self._post("/api/v1/outcomes", body)

    def run_evolution(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Trigger an evolution cycle.

        Parameters
        ----------
        thresholds : dict | None
            Optional threshold overrides.

        Returns
        -------
        dict
            Evolution report.
        """
        body: dict[str, Any] = {}
        if thresholds is not None:
            body["thresholds"] = thresholds
        return self._post("/api/v1/evolution", body)

    def get_health(self) -> dict[str, Any]:
        """Check server health.

        Returns
        -------
        dict
            Health status response.
        """
        return self._get("/api/v1/health")

    def get_dashboard(self) -> dict[str, Any]:
        """Get the dashboard summary.

        Returns
        -------
        dict
            Dashboard data (skill counts, health breakdown, etc.).
        """
        return self._get("/api/v1/dashboard")

    def get_analytics(self) -> dict[str, Any]:
        """Get the analytics report.

        Returns
        -------
        dict
            Analytics data (top skills, aggregate metrics, etc.).
        """
        return self._get("/api/v1/analytics")

    def get_conflicts(self) -> dict[str, Any]:
        """List potential skill conflicts.

        Returns
        -------
        dict
            Conflicts list and total count.
        """
        return self._get("/api/v1/conflicts")
