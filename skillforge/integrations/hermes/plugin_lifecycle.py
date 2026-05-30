"""SkillForge lifecycle hooks for Hermes Agent.

Hooks into agent lifecycle events to:
- Auto-import Hermes skills into SkillForge on session start
- Auto-record outcomes after skill-related tool calls
- Provide skill intelligence context to the agent

Supports dual-mode: local library or remote Docker API.
"""

from __future__ import annotations

import json
import logging
import sys
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Track which tools used skill_view / skill_manage so we can correlate
_last_skill_tool_id: str | None = None


def on_session_start(**kwargs) -> None:
    """Auto-import Hermes skills into SkillForge when a session begins.

    This ensures the Forge always has the latest skill data without
    requiring manual import. Silently succeeds — never blocks session
    start on import failures.

    Supports both local library and remote Docker API modes.
    """
    try:
        from plugins.skillforge.tools import _is_remote_mode, get_client, get_forge

        if _is_remote_mode():
            # Remote mode: use HTTP client to trigger import
            client = get_client()
            # The import reads from host filesystem via the handler
            from plugins.skillforge.tools import _handle_skillforge_import_remote
            result = _handle_skillforge_import_remote({})
            data = json.loads(result)
            imported = data.get("imported", 0)
            logger.info(
                "SkillForge session start (remote): imported %d Hermes skills",
                imported,
            )
        else:
            # Local mode: use library directly
            forge = get_forge()

            from hermes_constants import get_hermes_home
            from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter

            adapter = HermesSkillForgeAdapter(
                skillforge=forge._registry,
                hermes_skills_dir=str(get_hermes_home() / "skills"),
            )
            imported = adapter.import_hermes_skills()
            logger.info(
                "SkillForge session start (local): imported %d Hermes skills",
                len(imported),
            )
    except Exception:
        logger.warning(
            "SkillForge: failed to auto-import Hermes skills on session start",
            exc_info=True,
        )


def post_tool_call(**kwargs) -> Any:
    """Monitor skill-related tool calls for automatic outcome tracking.

    When the agent calls skill_view or skill_manage (read mode), we track
    the skill ID so that on a subsequent successful response we can
    automatically record the outcome.

    This hook doesn't modify tool results — it's an observer only.
    """
    global _last_skill_tool_id

    tool_name = kwargs.get("tool_name", "")
    tool_args = kwargs.get("tool_args", {})
    tool_result = kwargs.get("tool_result", "")

    # Track when skill_view is called
    if tool_name == "skill_view":
        skill_name = tool_args.get("name", "") if isinstance(tool_args, dict) else ""
        if skill_name:
            _last_skill_tool_id = f"hermes-{skill_name}"
            logger.debug("SkillForge tracking: skill_view(%s)", skill_name)

    # Track when skill_manage read is called
    elif tool_name == "skill_manage":
        action = tool_args.get("action", "") if isinstance(tool_args, dict) else ""
        if action in ("view", "list"):
            skill_name = tool_args.get("name", "") if isinstance(tool_args, dict) else ""
            if skill_name:
                _last_skill_tool_id = f"hermes-{skill_name}"

    # When a skills_list is called, track it for potential correlation
    elif tool_name == "skills_list":
        _last_skill_tool_id = "skills_list_query"

    # Return None — observer only, doesn't modify results
    return None


def get_tracked_skill_id() -> str | None:
    """Return the last tracked skill ID (for external callers)."""
    return _last_skill_tool_id


def clear_tracked_skill() -> None:
    """Clear the tracked skill ID after recording."""
    global _last_skill_tool_id
    _last_skill_tool_id = None
