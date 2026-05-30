"""SkillForge tool schemas and handlers for Hermes Agent.

Registers 5 tools into the 'skillforge' toolset:
- skillforge_load     — progressive skill search & loading
- skillforge_record   — record execution outcomes
- skillforge_evolve   — run evolution cycle
- skillforge_health   — health monitoring dashboard
- skillforge_import   — import Hermes SKILL.md skills into Forge

DUAL-MODE SUPPORT:
- LOCAL MODE (default): Uses bundled SkillForge library directly
- REMOTE MODE: Uses HTTP client to talk to SkillForge Docker container
  Set SKILLFORGE_API_URL env var (e.g. http://localhost:8742) to enable.
"""

from __future__ import annotations

import json
import logging
import sys
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode detection: local library vs remote Docker API
# ---------------------------------------------------------------------------

SKILLFORGE_API_URL = os.environ.get("SKILLFORGE_API_URL", "").strip()

def _is_remote_mode() -> bool:
    """Check if SkillForge is running as a remote Docker service."""
    return bool(SKILLFORGE_API_URL)


# ---------------------------------------------------------------------------
# Local mode setup (only when not remote)
# ---------------------------------------------------------------------------

if not _is_remote_mode():
    # Make the bundled skillforge package importable
    _plugin_dir = os.path.dirname(os.path.abspath(__file__))
    _sf_lib_dir = os.path.join(_plugin_dir, "_skillforge_lib")
    if _sf_lib_dir not in sys.path:
        sys.path.insert(0, _sf_lib_dir)

    from skillforge.forge import SkillForge
    from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter

# Re-export for other plugin modules
__all__ = ["get_forge", "close_forge", "get_client"]


# ---------------------------------------------------------------------------
# Singleton instances — shared by all tool handlers
# ---------------------------------------------------------------------------

_forge = None  # Local mode only
_client = None  # Remote mode only


def get_client():
    """Return the SkillForgeClient for remote mode. Creates lazily."""
    global _client
    if _client is None:
        # Add skillforge lib to path for client import
        _plugin_dir = os.path.dirname(os.path.abspath(__file__))
        _sf_lib_dir = os.path.join(_plugin_dir, "_skillforge_lib")
        if _sf_lib_dir not in sys.path:
            sys.path.insert(0, _sf_lib_dir)
        from skillforge.api.client import SkillForgeClient
        _client = SkillForgeClient(base_url=SKILLFORGE_API_URL, timeout=10.0)
        logger.info("SkillForge remote client: %s", SKILLFORGE_API_URL)
    return _client


def get_forge():
    """Return the shared SkillForge instance (local mode only)."""
    global _forge
    if _is_remote_mode():
        raise RuntimeError("Cannot use local forge in remote mode. Use get_client().")
    if _forge is None:
        from hermes_constants import get_hermes_home
        db_path = str(get_hermes_home() / "skillforge.db")
        logger.info("Initializing SkillForge (db=%s)", db_path)
        _forge = SkillForge(db_path=db_path)
    return _forge


def close_forge() -> None:
    """Close the shared SkillForge instance (called on shutdown)."""
    global _forge, _client
    if _forge is not None:
        _forge.close()
        _forge = None
    _client = None


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_skillforge_available() -> bool:
    """SkillForge is always available (pure Python, no external deps)."""
    return True


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SKILLFORGE_LOAD_SCHEMA = {
    "name": "skillforge_load",
    "description": (
        "Search and progressively load skills from the SkillForge intelligence "
        "platform. Returns skills ranked by Q-value (learned quality score), "
        "success rate, usage frequency, or relevance. Supports 3-tier "
        "progressive disclosure: tier 1 = metadata, tier 2 = core instructions, "
        "tier 3 = full resources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query or keywords to search for skills.",
            },
            "tier": {
                "type": "integer",
                "enum": [1, 2, 3],
                "description": "Detail level: 1 = metadata only, 2 = core instructions, 3 = full with resources. Default: 2.",
            },
            "routing": {
                "type": "string",
                "enum": ["q_value", "success_rate", "usage_count", "relevance"],
                "description": "Ranking strategy. Default: q_value.",
            },
            "limit": {
                "type": "integer",
                "description": "Max skills to return. Default: 5.",
            },
        },
        "required": ["query"],
    },
}

SKILLFORGE_RECORD_SCHEMA = {
    "name": "skillforge_record",
    "description": (
        "Record an execution outcome for a skill tracked by SkillForge. "
        "Updates Q-values via TD(lambda) learning, success rate, and usage "
        "count. Call this after completing a task that used a specific skill "
        "to help the system learn which skills work best."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "The SkillForge skill ID (e.g. 'hermes-axolotl' or a UUID).",
            },
            "success": {
                "type": "boolean",
                "description": "Whether the skill execution was successful.",
            },
            "latency_ms": {
                "type": "number",
                "description": "Wall-clock latency in milliseconds. Default: 0.",
            },
            "tokens_used": {
                "type": "integer",
                "description": "Number of LLM tokens consumed. Default: 0.",
            },
            "user_feedback": {
                "type": "number",
                "description": "Optional user feedback score on a 0-5 scale.",
            },
        },
        "required": ["skill_id", "success"],
    },
}

SKILLFORGE_EVOLVE_SCHEMA = {
    "name": "skillforge_evolve",
    "description": (
        "Run a full SkillForge evolution cycle across all tracked skills. "
        "Evaluates health, evolves under-performers (revises instructions), "
        "prunes dead skills, and returns a detailed evolution report. "
        "Use this periodically or when skill quality seems degraded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "q_critical": {
                "type": "number",
                "description": "Q-value threshold below which skills are critical (default: 0.25).",
            },
            "q_warning": {
                "type": "number",
                "description": "Q-value threshold below which skills get a warning (default: 0.40).",
            },
        },
    },
}

SKILLFORGE_HEALTH_SCHEMA = {
    "name": "skillforge_health",
    "description": (
        "Check the health status of skills tracked by SkillForge. "
        "Returns health scores based on Q-value, success rate, recency, "
        "and usage frequency. Can show a dashboard summary or individual "
        "skill health reports."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "Specific skill ID to check. If omitted, returns a dashboard summary of all skills.",
            },
            "show_recommendations": {
                "type": "boolean",
                "description": "Include actionable recommendations. Default: true.",
            },
        },
    },
}

SKILLFORGE_IMPORT_SCHEMA = {
    "name": "skillforge_import",
    "description": (
        "Import all Hermes Agent skills (from ~/.hermes/skills/) into the "
        "SkillForge registry. This scans every SKILL.md file, parses its "
        "frontmatter, and registers/updates the skill in the Forge with "
        "Q-value tracking. Run this after adding new Hermes skills to "
        "make them visible to SkillForge's intelligence layer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter (e.g. 'software-development'). If omitted, imports all.",
            },
            "force": {
                "type": "boolean",
                "description": "If true, re-import even if skills already exist (updates them). Default: true.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool handlers — REMOTE MODE (HTTP client)
# ---------------------------------------------------------------------------

def _handle_skillforge_load_remote(args: Dict[str, Any]) -> str:
    """Handle skillforge_load via HTTP API."""
    client = get_client()
    query = args.get("query", "")
    tier = args.get("tier", 2)
    routing = args.get("routing", "q_value")
    limit = args.get("limit", 5)

    try:
        result = client.load_skill(query=query, tier=tier, routing=routing, limit=limit)
        return json.dumps(result)
    except Exception as e:
        logger.exception("skillforge_load remote error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_record_remote(args: Dict[str, Any]) -> str:
    """Handle skillforge_record via HTTP API."""
    client = get_client()
    skill_id = args["skill_id"]
    success = args["success"]
    latency_ms = args.get("latency_ms")
    tokens_used = args.get("tokens_used")
    user_feedback = args.get("user_feedback")

    try:
        result = client.record_outcome(
            skill_id=skill_id,
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            user_feedback=user_feedback,
        )
        return json.dumps(result)
    except Exception as e:
        logger.exception("skillforge_record remote error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_evolve_remote(args: Dict[str, Any]) -> str:
    """Handle skillforge_evolve via HTTP API."""
    client = get_client()
    thresholds = {}
    if "q_critical" in args:
        thresholds["q_critical"] = args["q_critical"]
    if "q_warning" in args:
        thresholds["q_warning"] = args["q_warning"]

    try:
        result = client.run_evolution(thresholds=thresholds if thresholds else None)
        return json.dumps(result)
    except Exception as e:
        logger.exception("skillforge_evolve remote error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_health_remote(args: Dict[str, Any]) -> str:
    """Handle skillforge_health via HTTP API."""
    client = get_client()
    skill_id = args.get("skill_id")
    show_recommendations = args.get("show_recommendations", True)

    try:
        if skill_id:
            result = client.get_skill_stats(skill_id)
        else:
            result = client.get_dashboard()
        return json.dumps(result)
    except Exception as e:
        logger.exception("skillforge_health remote error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_import_remote(args: Dict[str, Any]) -> str:
    """Handle skillforge_import via HTTP API.

    NOTE: Remote import reads SKILL.md files from the HOST and sends them
    to the Docker API. The Docker container can't read ~/.hermes/skills/
    directly, so we parse locally and POST to the API.
    """
    client = get_client()

    try:
        # Read skills from host filesystem
        from hermes_constants import get_hermes_home
        skills_dir = get_hermes_home() / "skills"
        category = args.get("category")

        imported = []
        search_dir = skills_dir / category if category else skills_dir
        if not search_dir.is_dir():
            return json.dumps({"error": f"Directory not found: {search_dir}"})

        for skill_md in sorted(search_dir.rglob("SKILL.md")):
            try:
                content = skill_md.read_text(encoding="utf-8")
                # Parse frontmatter (reuse adapter's parser)
                _plugin_dir = os.path.dirname(os.path.abspath(__file__))
                _sf_lib_dir = os.path.join(_plugin_dir, "_skillforge_lib")
                if _sf_lib_dir not in sys.path:
                    sys.path.insert(0, _sf_lib_dir)
                from skillforge.integrations.hermes.adapter import (
                    _FRONTMATTER_RE, _parse_simple_yaml
                )

                match = _FRONTMATTER_RE.match(content)
                if not match:
                    continue
                frontmatter = _parse_simple_yaml(match.group(1))
                body = content[match.end():]

                name = str(frontmatter.get("name", "unnamed"))
                description = str(frontmatter.get("description", ""))
                skill_id = f"hermes-{name}"

                # Create or update skill via API
                try:
                    client.create_skill(
                        name=name,
                        description=description[:120],
                        instructions=body,
                        tags=["hermes_source"],
                        skill_id=skill_id,
                    )
                except Exception:
                    # Skill may already exist, try update
                    client.update_skill(skill_id, {
                        "tier1_metadata": description[:120],
                        "tier2_core": body,
                    })
                imported.append({"id": skill_id, "name": name})
            except Exception:
                logger.warning("Error importing %s", skill_md, exc_info=True)

        return json.dumps({"imported": len(imported), "skills": imported})
    except Exception as e:
        logger.exception("skillforge_import remote error")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool handlers — LOCAL MODE (direct library)
# ---------------------------------------------------------------------------

def _handle_skillforge_load_local(args: Dict[str, Any]) -> str:
    """Handle skillforge_load via local library."""
    forge = get_forge()
    query = args.get("query", "")
    tier = args.get("tier", 2)
    routing = args.get("routing", "q_value")
    limit = args.get("limit", 5)

    try:
        skills = forge.load_skill(
            query=query,
            tier=tier,
            routing=routing,
            limit=limit,
        )

        results = []
        for skill in skills:
            entry = {
                "id": skill.id,
                "name": skill.name,
                "tier1_metadata": skill.tier1_metadata,
                "q_value": round(skill.q_value, 4),
                "success_rate": round(skill.success_rate, 4),
                "usage_count": skill.usage_count,
                "lifecycle": skill.lifecycle.value,
                "tags": skill.tags,
            }
            if tier >= 2 and skill.tier2_core:
                entry["instructions"] = skill.tier2_core[:2000]  # Truncate for context
            if tier >= 3 and skill.tier3_resources:
                entry["resources"] = skill.tier3_resources
            results.append(entry)

        return json.dumps({
            "count": len(results),
            "tier": tier,
            "routing": routing,
            "skills": results,
        })
    except Exception as e:
        logger.exception("skillforge_load error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_record_local(args: Dict[str, Any]) -> str:
    """Handle skillforge_record via local library."""
    forge = get_forge()
    skill_id = args["skill_id"]
    success = args["success"]
    latency_ms = args.get("latency_ms")
    tokens_used = args.get("tokens_used")
    user_feedback = args.get("user_feedback")

    try:
        forge.record_outcome(
            skill_id=skill_id,
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            user_feedback=user_feedback,
        )

        # Return updated stats
        stats = forge.get_skill_stats(skill_id)
        return json.dumps({
            "recorded": True,
            "skill_id": skill_id,
            "success": success,
            "updated_stats": stats,
        })
    except Exception as e:
        logger.exception("skillforge_record error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_evolve_local(args: Dict[str, Any]) -> str:
    """Handle skillforge_evolve via local library."""
    forge = get_forge()

    thresholds = {}
    if "q_critical" in args:
        thresholds["q_critical"] = args["q_critical"]
    if "q_warning" in args:
        thresholds["q_warning"] = args["q_warning"]

    try:
        report = forge.run_evolution_loop(
            thresholds=thresholds if thresholds else None,
        )
        return json.dumps({
            "evolution_complete": True,
            "report": report.to_dict(),
            "summary": report.summary(),
        })
    except Exception as e:
        logger.exception("skillforge_evolve error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_health_local(args: Dict[str, Any]) -> str:
    """Handle skillforge_health via local library."""
    forge = get_forge()
    skill_id = args.get("skill_id")
    show_recommendations = args.get("show_recommendations", True)

    try:
        if skill_id:
            # Individual skill health
            health_monitor = _get_health_monitor(forge)
            report = health_monitor.check_health(skill_id)
            result = {
                "skill_id": report.skill_id,
                "status": report.status.value,
                "health_score": round(report.health_score, 4),
                "q_value": round(report.q_value, 4),
                "success_rate": round(report.success_rate, 4),
                "usage_count": report.usage_count,
                "last_used": report.last_used,
                "days_since_last_use": round(report.days_since_last_use, 1),
                "failure_rate": round(report.failure_rate, 4),
            }
            if show_recommendations:
                result["recommendations"] = report.recommendations
            return json.dumps(result)
        else:
            # Dashboard summary
            health_monitor = _get_health_monitor(forge)
            dashboard = health_monitor.get_dashboard_summary()
            if show_recommendations:
                # Add worst-skill recommendations
                all_reports = health_monitor.check_all()
                for r in all_reports[:3]:
                    if r.recommendations:
                        dashboard.setdefault("top_issues", {})
                        dashboard["top_issues"][r.skill_id] = {
                            "health_score": round(r.health_score, 4),
                            "status": r.status.value,
                            "recommendations": r.recommendations,
                        }
            return json.dumps(dashboard)
    except Exception as e:
        logger.exception("skillforge_health error")
        return json.dumps({"error": str(e)})


def _handle_skillforge_import_local(args: Dict[str, Any]) -> str:
    """Handle skillforge_import via local library."""
    forge = get_forge()

    try:
        # Build adapter pointing at Hermes skills directory
        from hermes_constants import get_hermes_home
        adapter = HermesSkillForgeAdapter(
            skillforge=forge._registry,
            hermes_skills_dir=str(get_hermes_home() / "skills"),
        )

        # If category specified, filter
        category = args.get("category")
        if category:
            skills_dir = get_hermes_home() / "skills" / category
            if not skills_dir.is_dir():
                return json.dumps({
                    "error": f"Category directory not found: {category}",
                    "path": str(skills_dir),
                })
            # Import from specific category only
            imported = []
            for skill_md in sorted(skills_dir.rglob("SKILL.md")):
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    parsed = adapter._parse_skill_md(content)
                    if parsed is None:
                        continue
                    skill = adapter._register_parsed_skill(parsed)
                    imported.append(skill)
                except Exception:
                    logger.warning("Error importing %s", skill_md, exc_info=True)
        else:
            imported = adapter.import_hermes_skills()

        return json.dumps({
            "imported": len(imported),
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "lifecycle": s.lifecycle.value,
                }
                for s in imported
            ],
        })
    except Exception as e:
        logger.exception("skillforge_import error")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Unified dispatchers — route to local or remote handler
# ---------------------------------------------------------------------------

def _handle_skillforge_load(args: Dict[str, Any], **kwargs) -> str:
    if _is_remote_mode():
        return _handle_skillforge_load_remote(args)
    return _handle_skillforge_load_local(args)


def _handle_skillforge_record(args: Dict[str, Any], **kwargs) -> str:
    if _is_remote_mode():
        return _handle_skillforge_record_remote(args)
    return _handle_skillforge_record_local(args)


def _handle_skillforge_evolve(args: Dict[str, Any], **kwargs) -> str:
    if _is_remote_mode():
        return _handle_skillforge_evolve_remote(args)
    return _handle_skillforge_evolve_local(args)


def _handle_skillforge_health(args: Dict[str, Any], **kwargs) -> str:
    if _is_remote_mode():
        return _handle_skillforge_health_remote(args)
    return _handle_skillforge_health_local(args)


def _handle_skillforge_import(args: Dict[str, Any], **kwargs) -> str:
    if _is_remote_mode():
        return _handle_skillforge_import_remote(args)
    return _handle_skillforge_import_local(args)


# ---------------------------------------------------------------------------
# Health monitor (lazy singleton — local mode only)
# ---------------------------------------------------------------------------

_health_monitor = None


def _get_health_monitor(forge):
    """Get or create the HealthMonitor singleton."""
    global _health_monitor
    if _health_monitor is None:
        from skillforge.intelligence.health_monitor import HealthMonitor
        _health_monitor = HealthMonitor(
            registry=forge._registry,
            tracker=forge._tracker,
            graph=forge._graph,
        )
    return _health_monitor
