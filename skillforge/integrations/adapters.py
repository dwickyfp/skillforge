"""Integration adapters for connecting SkillForge with external platforms.

Adapters provide a unified interface for triggering actions on external
systems in response to SkillForge events (skill creation, experiment
results, etc.) and for ingesting data from external sources.

Each adapter implements the :class:`IntegrationAdapter` protocol and is
registered with an :class:`AdapterRegistry` for discovery and dispatch.

Currently implemented adapters:

- :class:`GitHubAdapter` — syncs skills with a GitHub repository via
  the REST API (uses ``urllib``, no ``requests`` dependency).
- :class:`SlackAdapter` — posts notifications to a Slack channel via
  an incoming webhook.
- :class:`WebhookAdapter` — generic HTTP webhook adapter for any
  custom endpoint.
- :class:`FileAdapter` — exports skills as JSON or markdown files to
  a local directory.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AdapterStatus(Enum):
    """Operational status of an adapter."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"


class EventType(Enum):
    """Types of SkillForge events that adapters can react to."""

    SKILL_CREATED = "skill_created"
    SKILL_UPDATED = "skill_updated"
    SKILL_DELETED = "skill_deleted"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    METRIC_ALERT = "metric_alert"
    SYNC_REQUEST = "sync_request"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """Configuration for an adapter.

    Attributes
    ----------
    name : str
        Unique adapter name.
    adapter_type : str
        Type identifier (e.g. ``"github"``, ``"slack"``).
    enabled : bool
        Whether this adapter is active.
    config : dict[str, Any]
        Adapter-specific configuration (URLs, tokens, etc.).
    events : list[EventType]
        Which events trigger this adapter.
    """

    name: str
    adapter_type: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    events: list[EventType] = field(default_factory=list)


@dataclass
class EventPayload:
    """Data payload for an adapter event.

    Attributes
    ----------
    event_type : EventType
        The type of event.
    data : dict[str, Any]
        Event-specific data (skill_id, experiment_id, metrics, etc.).
    timestamp : str
        ISO-8601 UTC timestamp of the event.
    source : str
        Identifier of the component that emitted the event.
    """

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source: str = "skillforge"

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AdapterResult:
    """Result of an adapter invocation.

    Attributes
    ----------
    adapter_name : str
        Name of the adapter that was invoked.
    success : bool
        Whether the operation succeeded.
    message : str
        Human-readable status message.
    response_data : dict[str, Any]
        Any data returned by the external system.
    duration_ms : float
        Wall-clock time in milliseconds.
    """

    adapter_name: str
    success: bool
    message: str = ""
    response_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IntegrationAdapter(Protocol):
    """Protocol that all integration adapters must satisfy."""

    @property
    def name(self) -> str:
        """Unique adapter name."""
        ...

    @property
    def status(self) -> AdapterStatus:
        """Current operational status."""
        ...

    def dispatch(self, event: EventPayload) -> AdapterResult:
        """Handle an event and return the result."""
        ...

    def health_check(self) -> bool:
        """Return whether the adapter can reach its external system."""
        ...


# ---------------------------------------------------------------------------
# HTTP helper (stdlib urllib)
# ---------------------------------------------------------------------------


def _http_request(
    url: str,
    method: str = "POST",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, str]:
    """Execute an HTTP request using stdlib urllib.

    Parameters
    ----------
    url : str
        The target URL.
    method : str
        HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``).
    body : bytes | None
        Request body (e.g. JSON-encoded).
    headers : dict[str, str] | None
        HTTP headers.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    tuple[int, str]
        ``(status_code, response_body)``.
    """
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8", errors="replace")
            return resp.status, response_body
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, response_body
    except urllib.error.URLError as exc:
        raise ConnectionError(f"HTTP request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"HTTP request timed out after {timeout}s") from exc


# ---------------------------------------------------------------------------
# GitHub adapter
# ---------------------------------------------------------------------------


class GitHubAdapter:
    """Sync skills with a GitHub repository using the REST API.

    Parameters
    ----------
    token : str
        GitHub personal access token (PAT).
    repo : str
        Repository in ``owner/name`` format.
    branch : str
        Target branch for file updates.
    base_url : str
        Base API URL (default ``https://api.github.com``).  Override for
        GitHub Enterprise.
    """

    def __init__(
        self,
        token: str,
        repo: str,
        branch: str = "main",
        base_url: str = "https://api.github.com",
    ) -> None:
        self._token = token
        self._repo = repo
        self._branch = branch
        self._base_url = base_url.rstrip("/")
        self._status = AdapterStatus.ACTIVE
        self._name = f"github:{repo}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AdapterStatus:
        return self._status

    def dispatch(self, event: EventPayload) -> AdapterResult:
        """Push a skill file to GitHub on creation/update events.

        For ``SKILL_CREATED`` and ``SKILL_UPDATED`` events, expects
        ``event.data`` to contain ``path`` (repository file path),
        ``content`` (base64-encoded file content), and optionally
        ``commit_message``.
        """
        import base64
        import time

        start = time.monotonic()
        data = event.data
        path = data.get("path", "")
        content = data.get("content", "")

        if not path or not content:
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message="Missing 'path' or 'content' in event data",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Ensure content is base64-encoded
        if not data.get("is_base64", False):
            content = base64.b64encode(content.encode("utf-8")).decode("ascii")

        url = (
            f"{self._base_url}/repos/{self._repo}/contents/{path}"
            f"?ref={self._branch}"
        )
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # Try to get existing file SHA for update
        sha: str | None = None
        try:
            _, resp_body = _http_request(url, method="GET", headers=headers, timeout=15)
            existing = json.loads(resp_body)
            sha = existing.get("sha")
        except (ConnectionError, json.JSONDecodeError):
            pass  # File doesn't exist yet — will create

        body: dict[str, Any] = {
            "message": data.get(
                "commit_message", f"SkillForge: update {path}"
            ),
            "content": content,
            "branch": self._branch,
        }
        if sha:
            body["sha"] = sha

        try:
            status_code, resp_body = _http_request(
                url,
                method="PUT",
                body=json.dumps(body).encode("utf-8"),
                headers=headers,
            )
            elapsed = (time.monotonic() - start) * 1000
            if status_code in (200, 201):
                resp_data = json.loads(resp_body) if resp_body else {}
                return AdapterResult(
                    adapter_name=self._name,
                    success=True,
                    message=f"Pushed {path} to {self._repo}",
                    response_data=resp_data,
                    duration_ms=elapsed,
                )
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"GitHub API returned {status_code}: {resp_body[:200]}",
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Error: {exc}",
                duration_ms=elapsed,
            )

    def health_check(self) -> bool:
        """Check if the GitHub API is reachable and token is valid."""
        url = f"{self._base_url}/user"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }
        try:
            status_code, _ = _http_request(url, method="GET", headers=headers, timeout=10)
            ok = status_code == 200
            self._status = AdapterStatus.ACTIVE if ok else AdapterStatus.ERROR
            return ok
        except Exception:
            self._status = AdapterStatus.ERROR
            return False


# ---------------------------------------------------------------------------
# Slack adapter
# ---------------------------------------------------------------------------


class SlackAdapter:
    """Post notifications to a Slack channel via an incoming webhook.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL.
    channel : str | None
        Override the default webhook channel.
    username : str
        Bot username for messages.
    icon_emoji : str
        Emoji avatar for the bot.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "SkillForge",
        icon_emoji: str = ":robot_face:",
    ) -> None:
        self._webhook_url = webhook_url
        self._channel = channel
        self._username = username
        self._icon_emoji = icon_emoji
        self._status = AdapterStatus.ACTIVE
        self._name = "slack:webhook"

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AdapterStatus:
        return self._status

    def dispatch(self, event: EventPayload) -> AdapterResult:
        """Format the event as a Slack message and post it."""
        import time

        start = time.monotonic()
        message = self._format_message(event)
        payload: dict[str, Any] = {
            "text": message,
            "username": self._username,
            "icon_emoji": self._icon_emoji,
        }
        if self._channel:
            payload["channel"] = self._channel

        try:
            status_code, body = _http_request(
                self._webhook_url,
                method="POST",
                body=json.dumps(payload).encode("utf-8"),
                timeout=15,
            )
            elapsed = (time.monotonic() - start) * 1000
            if status_code == 200 and body.strip() == "ok":
                return AdapterResult(
                    adapter_name=self._name,
                    success=True,
                    message="Posted to Slack",
                    duration_ms=elapsed,
                )
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Slack returned {status_code}: {body[:200]}",
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Error: {exc}",
                duration_ms=elapsed,
            )

    def health_check(self) -> bool:
        """Send a test message to verify the webhook is alive."""
        try:
            payload = {
                "text": "🔬 SkillForge health check",
                "username": self._username,
                "icon_emoji": self._icon_emoji,
            }
            if self._channel:
                payload["channel"] = self._channel
            status_code, body = _http_request(
                self._webhook_url,
                method="POST",
                body=json.dumps(payload).encode("utf-8"),
                timeout=10,
            )
            ok = status_code == 200 and body.strip() == "ok"
            self._status = AdapterStatus.ACTIVE if ok else AdapterStatus.ERROR
            return ok
        except Exception:
            self._status = AdapterStatus.ERROR
            return False

    def _format_message(self, event: EventPayload) -> str:
        """Format an event into a human-readable Slack message."""
        emoji_map = {
            EventType.SKILL_CREATED: "🆕",
            EventType.SKILL_UPDATED: "🔄",
            EventType.SKILL_DELETED: "🗑️",
            EventType.EXPERIMENT_STARTED: "🧪",
            EventType.EXPERIMENT_COMPLETED: "📊",
            EventType.METRIC_ALERT: "⚠️",
            EventType.SYNC_REQUEST: "🔁",
        }
        emoji = emoji_map.get(event.event_type, "📋")
        data = event.data
        lines: list[str] = []

        if event.event_type in (EventType.SKILL_CREATED, EventType.SKILL_UPDATED):
            skill_name = data.get("name", "Unknown")
            skill_id = data.get("skill_id", "")
            lines.append(
                f"{emoji} *{event.event_type.value.replace('_', ' ').title()}*: "
                f"`{skill_name}` (ID: `{skill_id}`)"
            )
        elif event.event_type == EventType.EXPERIMENT_COMPLETED:
            exp_name = data.get("name", "Unknown")
            result = data.get("summary", "No summary available")
            lines.append(
                f"{emoji} *Experiment Completed*: `{exp_name}`\n{result}"
            )
        elif event.event_type == EventType.METRIC_ALERT:
            metric = data.get("metric", "unknown")
            value = data.get("value", "?")
            lines.append(f"{emoji} *Metric Alert*: {metric} = {value}")
        else:
            lines.append(f"{emoji} *{event.event_type.value}*")

        if event.source:
            lines.append(f"_Source: {event.source}_")
        lines.append(f"_{event.timestamp}_")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Webhook adapter (generic)
# ---------------------------------------------------------------------------


class WebhookAdapter:
    """Generic webhook adapter that POSTs events to an arbitrary URL.

    Parameters
    ----------
    name : str
        Adapter name.
    url : str
        Target webhook URL.
    secret : str | None
        Optional shared secret for HMAC signature verification on the
        receiving end.
    headers : dict[str, str]
        Additional headers to include in every request.
    """

    def __init__(
        self,
        name: str,
        url: str,
        secret: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._name = name
        self._url = url
        self._secret = secret
        self._extra_headers = headers or {}
        self._status = AdapterStatus.ACTIVE

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AdapterStatus:
        return self._status

    def dispatch(self, event: EventPayload) -> AdapterResult:
        """POST the event as JSON to the webhook URL."""
        import time

        start = time.monotonic()
        payload = {
            "event_type": event.event_type.value,
            "data": event.data,
            "timestamp": event.timestamp,
            "source": event.source,
        }

        # Optional HMAC signature
        body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        headers = dict(self._extra_headers)

        if self._secret:
            import hashlib
            import hmac as hmac_mod

            signature = hmac_mod.new(
                self._secret.encode("utf-8"),
                body_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-SkillForge-Signature"] = f"sha256={signature}"

        try:
            status_code, resp_body = _http_request(
                self._url,
                method="POST",
                body=body_bytes,
                headers=headers,
                timeout=15,
            )
            elapsed = (time.monotonic() - start) * 1000
            if 200 <= status_code < 300:
                resp_data: dict[str, Any] = {}
                try:
                    resp_data = json.loads(resp_body)
                except json.JSONDecodeError:
                    pass
                return AdapterResult(
                    adapter_name=self._name,
                    success=True,
                    message=f"Webhook responded with {status_code}",
                    response_data=resp_data,
                    duration_ms=elapsed,
                )
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Webhook returned {status_code}: {resp_body[:200]}",
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Error: {exc}",
                duration_ms=elapsed,
            )

    def health_check(self) -> bool:
        """Send a minimal ``GET`` to verify the endpoint is reachable."""
        try:
            status_code, _ = _http_request(
                self._url,
                method="GET",
                headers=self._extra_headers,
                timeout=5,
            )
            self._status = AdapterStatus.ACTIVE
            return True
        except Exception:
            self._status = AdapterStatus.ERROR
            return False


# ---------------------------------------------------------------------------
# File adapter (local export)
# ---------------------------------------------------------------------------


class FileAdapter:
    """Export skills/events as JSON or markdown files to a local directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write files to (created if missing).
    format : str
        Output format — ``"json"`` or ``"markdown"``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        format: str = "json",
    ) -> None:
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._format = format.lower()
        self._status = AdapterStatus.ACTIVE
        self._name = f"file:{self._output_dir.name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AdapterStatus:
        return self._status

    def dispatch(self, event: EventPayload) -> AdapterResult:
        """Write event data to a file in the output directory."""
        import time

        start = time.monotonic()
        data = event.data

        # Derive filename
        skill_id = data.get("skill_id", data.get("name", "event"))
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(skill_id))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if self._format == "markdown":
            filename = f"{safe_name}_{timestamp}.md"
            content = self._to_markdown(event)
        else:
            filename = f"{safe_name}_{timestamp}.json"
            content = json.dumps(
                {
                    "event_type": event.event_type.value,
                    "data": data,
                    "timestamp": event.timestamp,
                    "source": event.source,
                },
                indent=2,
                ensure_ascii=False,
            )

        filepath = self._output_dir / filename
        try:
            filepath.write_text(content, encoding="utf-8")
            elapsed = (time.monotonic() - start) * 1000
            return AdapterResult(
                adapter_name=self._name,
                success=True,
                message=f"Wrote {filepath}",
                response_data={"path": str(filepath)},
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return AdapterResult(
                adapter_name=self._name,
                success=False,
                message=f"Error writing file: {exc}",
                duration_ms=elapsed,
            )

    def health_check(self) -> bool:
        """Verify the output directory is writable."""
        try:
            test_file = self._output_dir / ".skillforge_health_check"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            self._status = AdapterStatus.ACTIVE
            return True
        except Exception:
            self._status = AdapterStatus.ERROR
            return False

    @staticmethod
    def _to_markdown(event: EventPayload) -> str:
        """Convert an event to a simple markdown document."""
        data = event.data
        lines = [
            f"# {event.event_type.value.replace('_', ' ').title()}",
            "",
            f"**Source:** {event.source}  ",
            f"**Timestamp:** {event.timestamp}",
            "",
            "## Data",
            "",
        ]
        for key, value in data.items():
            lines.append(f"- **{key}:** `{value}`")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------


class AdapterRegistry:
    """Central registry for managing integration adapters.

    Provides registration, lookup, and event dispatching across all
    configured adapters.

    Examples
    --------
    >>> registry = AdapterRegistry()
    >>> registry.register(FileAdapter("/tmp/skills"))
    >>> registry.dispatch(EventPayload(EventType.SKILL_CREATED, {"name": "test"}))
    """

    def __init__(self) -> None:
        self._adapters: dict[str, IntegrationAdapter] = {}

    @property
    def adapter_names(self) -> list[str]:
        """Return names of all registered adapters."""
        return list(self._adapters.keys())

    @property
    def adapter_count(self) -> int:
        """Return the number of registered adapters."""
        return len(self._adapters)

    def register(self, adapter: IntegrationAdapter) -> None:
        """Register an adapter.

        Parameters
        ----------
        adapter : IntegrationAdapter
            The adapter instance to register.

        Raises
        ------
        TypeError
            If *adapter* does not satisfy the
            :class:`IntegrationAdapter` protocol.
        """
        if not isinstance(adapter, IntegrationAdapter):
            raise TypeError(
                f"Object {type(adapter).__name__} does not implement "
                f"IntegrationAdapter protocol"
            )
        self._adapters[adapter.name] = adapter
        logger.info("Registered adapter: %s", adapter.name)

    def unregister(self, name: str) -> bool:
        """Unregister an adapter by name.

        Returns
        -------
        bool
            ``True`` if the adapter was found and removed.
        """
        if name in self._adapters:
            del self._adapters[name]
            logger.info("Unregistered adapter: %s", name)
            return True
        return False

    def get(self, name: str) -> IntegrationAdapter | None:
        """Retrieve an adapter by name."""
        return self._adapters.get(name)

    def dispatch(
        self,
        event: EventPayload,
        target_adapters: list[str] | None = None,
    ) -> list[AdapterResult]:
        """Dispatch an event to registered adapters.

        Parameters
        ----------
        event : EventPayload
            The event to dispatch.
        target_adapters : list[str] | None
            If provided, only these adapters receive the event.
            Otherwise all registered adapters are notified.

        Returns
        -------
        list[AdapterResult]
            Results from each adapter.
        """
        results: list[AdapterResult] = []
        adapters = (
            [self._adapters[n] for n in target_adapters if n in self._adapters]
            if target_adapters
            else list(self._adapters.values())
        )

        for adapter in adapters:
            try:
                result = adapter.dispatch(event)
                results.append(result)
                if not result.success:
                    logger.warning(
                        "Adapter '%s' failed: %s",
                        adapter.name,
                        result.message,
                    )
            except Exception as exc:
                logger.exception("Adapter '%s' raised exception", adapter.name)
                results.append(
                    AdapterResult(
                        adapter_name=adapter.name,
                        success=False,
                        message=f"Unhandled exception: {exc}",
                    )
                )

        return results

    def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all registered adapters.

        Returns
        -------
        dict[str, bool]
            Mapping of adapter name to health status.
        """
        results: dict[str, bool] = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = adapter.health_check()
            except Exception:
                logger.exception("Health check failed for '%s'", name)
                results[name] = False
        return results

    def list_adapters(
        self, status: AdapterStatus | None = None
    ) -> list[dict[str, Any]]:
        """List registered adapters with their status.

        Parameters
        ----------
        status : AdapterStatus | None
            If provided, filter by this status.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has keys ``name``, ``type``, ``status``.
        """
        adapters: list[dict[str, Any]] = []
        for adapter in self._adapters.values():
            if status is not None and adapter.status != status:
                continue
            adapters.append(
                {
                    "name": adapter.name,
                    "type": type(adapter).__name__,
                    "status": adapter.status.value,
                }
            )
        return adapters

    def dispatch_to_file(
        self,
        event: EventPayload,
        target_adapters: list[str] | None = None,
    ) -> dict[str, Any]:
        """Dispatch an event and return a serialisable summary.

        Useful for programmatic consumption rather than human logging.

        Returns
        -------
        dict[str, Any]
            Summary with ``total``, ``succeeded``, ``failed``, and
            ``results`` keys.
        """
        results = self.dispatch(event, target_adapters)
        return {
            "total": len(results),
            "succeeded": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [
                {
                    "adapter": r.adapter_name,
                    "success": r.success,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }
