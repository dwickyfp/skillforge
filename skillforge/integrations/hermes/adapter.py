"""Bidirectional adapter between SkillForge and Hermes Agent skills.

Reads Hermes ``SKILL.md`` files (YAML frontmatter + markdown body) and
synchronises them with the SkillForge registry.  Also exports SkillForge
skills back to ``SKILL.md`` format for consumption by Hermes Agent.

SKILL.md format (agentskills.io standard)::

    ---
    name: my-skill
    description: "Short description."
    version: 1.0.0
    author: someone
    metadata:
      hermes:
        tags: [tag1, tag2]
    ---
    # My Skill
    Body content …
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.registry import Skill, SkillLifecycle, SkillRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML frontmatter parser (stdlib-only, no PyYAML dependency)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a simple YAML document into a Python dict.

    Handles flat key-value pairs, nested keys (one level), inline lists
    ``[a, b]``, and quoted strings.  Does **not** attempt full YAML
    compliance — just enough for SKILL.md frontmatter.
    """
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())

        if indent == 0 and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_section = None
            current_key = key
            if value:
                result[key] = _coerce_yaml_value(value)
            else:
                result[key] = {}
                current_section = result[key]

        elif indent >= 2 and current_key and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if current_section is not None:
                current_section[key] = _coerce_yaml_value(value)
            else:
                # Nested under a key that already has a scalar value
                nested: dict[str, Any] = {key: _coerce_yaml_value(value)}
                result[current_key] = nested
                current_section = nested

        elif indent >= 4 and current_section is not None and ":" in line:
            key, _, value = line.partition(":")
            current_section[key.strip()] = _coerce_yaml_value(value.strip())

    return result


def _coerce_yaml_value(value: str) -> Any:
    """Coerce a YAML scalar to a Python value."""
    if not value:
        return ""
    # Inline list: [a, b, c]
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [_coerce_yaml_value(item.strip()) for item in items if item.strip()]
    # Quoted string
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # Null
    if value.lower() in ("null", "~"):
        return None
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _generate_simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """Serialise a simple dict to YAML-ish text (for SKILL.md frontmatter)."""
    lines: list[str] = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_generate_simple_yaml(value, indent + 1))
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value)
            lines.append(f"{prefix}{key}: [{items}]")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f'{prefix}{key}: "{value}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HermesSkillForgeAdapter:
    """Bidirectional bridge between Hermes Agent skills and SkillForge.

    Parameters
    ----------
    skillforge : SkillRegistry
        The SkillForge registry to synchronise with.
    hermes_skills_dir : str
        Path to the Hermes skills root directory.  Defaults to
        ``~/.hermes/skills``.
    """

    def __init__(
        self,
        skillforge: SkillRegistry,
        hermes_skills_dir: str | Path = "~/.hermes/skills",
    ) -> None:
        # Accept either a SkillForge instance (which wraps SkillRegistry
        # in ._registry) or a SkillRegistry directly.
        if hasattr(skillforge, '_registry') and not isinstance(
            skillforge, SkillRegistry
        ):
            self._registry = skillforge._registry
        else:
            self._registry = skillforge
        self._skills_dir = Path(hermes_skills_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Import: Hermes → SkillForge
    # ------------------------------------------------------------------

    def import_hermes_skills(self) -> list[Skill]:
        """Discover all ``SKILL.md`` files under the Hermes skills directory
        and register (or update) them in the SkillForge registry.

        Returns
        -------
        list[Skill]
            The imported / updated ``Skill`` objects.
        """
        imported: list[Skill] = []

        if not self._skills_dir.is_dir():
            logger.warning(
                "Hermes skills directory does not exist: %s", self._skills_dir
            )
            return imported

        for skill_md in sorted(self._skills_dir.rglob("SKILL.md")):
            try:
                content = skill_md.read_text(encoding="utf-8")
                parsed = self._parse_skill_md(content)
                if parsed is None:
                    logger.warning("Failed to parse %s — skipping", skill_md)
                    continue
                skill = self._register_parsed_skill(parsed)
                imported.append(skill)
            except Exception:
                logger.exception("Error importing %s", skill_md)

        logger.info("Imported %d Hermes skills into SkillForge", len(imported))
        return imported

    # ------------------------------------------------------------------
    # Export: SkillForge → Hermes
    # ------------------------------------------------------------------

    def export_skill_to_hermes(
        self,
        skill_id: str,
        category: str = "custom",
    ) -> Path | None:
        """Convert a SkillForge skill into a ``SKILL.md`` file and write it
        to the Hermes skills directory under *category*.

        Parameters
        ----------
        skill_id : str
            The SkillForge skill ID to export.
        category : str
            The Hermes category directory (e.g. ``productivity``).

        Returns
        -------
        Path | None
            Path to the written ``SKILL.md``, or ``None`` if the skill was
            not found.
        """
        skill = self._registry.get_skill(skill_id, tier=3)
        if skill is None:
            logger.warning("Skill '%s' not found — cannot export", skill_id)
            return None

        md_content = self._generate_skill_md(skill)

        dest_dir = self._skills_dir / category / skill.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "SKILL.md"
        dest_path.write_text(md_content, encoding="utf-8")

        logger.info("Exported skill '%s' → %s", skill_id, dest_path)
        return dest_path

    # ------------------------------------------------------------------
    # Sync: bidirectional
    # ------------------------------------------------------------------

    def sync(self) -> dict[str, Any]:
        """Perform a bidirectional synchronisation between Hermes skills
        and the SkillForge registry.

        1. **Import** all Hermes skills into SkillForge (upsert).
        2. **Export** any SkillForge skills that were originally imported
           from Hermes but are missing from disk.

        Returns
        -------
        dict
            Summary with keys ``imported``, ``exported``, ``errors``.
        """
        summary: dict[str, Any] = {
            "imported": 0,
            "exported": 0,
            "errors": [],
        }

        # Phase 1: import
        try:
            imported = self.import_hermes_skills()
            summary["imported"] = len(imported)
        except Exception as exc:
            summary["errors"].append(f"Import phase error: {exc}")

        # Phase 2: export skills tagged with hermes_source
        try:
            all_skills = self._registry.list_skills(limit=10000)
            for skill in all_skills:
                if "hermes_source" in skill.tags:
                    category = "skillforge-export"
                    md_path = self._skills_dir / category / skill.name / "SKILL.md"
                    if not md_path.exists():
                        result = self.export_skill_to_hermes(skill.id, category)
                        if result is not None:
                            summary["exported"] += 1
        except Exception as exc:
            summary["errors"].append(f"Export phase error: {exc}")

        logger.info(
            "Sync complete — imported: %d, exported: %d, errors: %d",
            summary["imported"],
            summary["exported"],
            len(summary["errors"]),
        )
        return summary

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_skill_md(self, content: str) -> dict[str, Any] | None:
        """Parse a Hermes ``SKILL.md`` file into structured data.

        Returns a dict with keys ``frontmatter`` (parsed YAML dict) and
        ``body`` (markdown string), or ``None`` if parsing fails.
        """
        match = _FRONTMATTER_RE.match(content)
        if not match:
            logger.debug("No YAML frontmatter found")
            return None

        raw_yaml = match.group(1)
        frontmatter = _parse_simple_yaml(raw_yaml)

        # Validate required fields
        if "name" not in frontmatter:
            logger.debug("Frontmatter missing required 'name' field")
            return None

        body = content[match.end():]
        return {"frontmatter": frontmatter, "body": body}

    def _register_parsed_skill(self, parsed: dict[str, Any]) -> Skill:
        """Register or update a parsed SKILL.md in the SkillForge registry."""
        fm = parsed["frontmatter"]
        body = parsed["body"]

        name = str(fm.get("name", "unnamed"))
        description = str(fm.get("description", ""))
        version_str = str(fm.get("version", "1.0.0"))

        # Extract tags from metadata.hermes.tags or generate from name
        tags: list[str] = []
        metadata = fm.get("metadata", {})
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {})
            if isinstance(hermes_meta, dict):
                raw_tags = hermes_meta.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [str(t) for t in raw_tags]

        tags.append("hermes_source")

        # Deterministic ID based on name to allow upsert
        skill_id = f"hermes-{name}"

        # Check if already registered
        existing = self._registry.get_skill(skill_id, tier=3)
        if existing is not None:
            # Update existing
            updates: dict[str, Any] = {
                "name": name,
                "tier1_metadata": description[:120],
                "tier2_core": body,
                "tags": tags,
            }
            # Bump version if body changed
            if existing.tier2_core != body:
                updates["version"] = existing.version + 1
            updated = self._registry.update_skill(skill_id, updates)
            assert updated is not None
            logger.debug("Updated Hermes skill: %s", name)
            return updated

        # Register new
        skill = self._registry.register_skill(
            name=name,
            tier1_metadata=description[:120],
            tier2_core=body,
            tags=tags,
            skill_id=skill_id,
        )
        # Set to ACTIVE since these are curated skills
        self._registry.update_skill(skill.id, {"lifecycle": SkillLifecycle.ACTIVE.value})
        skill = self._registry.get_skill(skill.id, tier=3)
        assert skill is not None
        logger.debug("Registered new Hermes skill: %s", name)
        return skill

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_skill_md(self, skill: Skill) -> str:
        """Generate a ``SKILL.md`` string from a :class:`Skill` object."""
        frontmatter: dict[str, Any] = {
            "name": skill.name,
            "description": skill.tier1_metadata,
            "version": f"{skill.version}.0.0",
            "author": "skillforge",
            "metadata": {
                "hermes": {
                    "tags": [t for t in skill.tags if t != "hermes_source"],
                },
            },
        }

        yaml_text = _generate_simple_yaml(frontmatter)
        return f"---\n{yaml_text}\n---\n\n{skill.tier2_core}\n"
