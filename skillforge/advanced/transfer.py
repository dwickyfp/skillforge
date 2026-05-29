"""Cross-agent skill transfer engine.

Exports skills to and imports skills from various agent formats:
- **Hermes**: YAML frontmatter + markdown body (SKILL.md)
- **OpenClaw**: ``.learnings/`` structured files
- **JSON**: flat JSON array
- **Markdown**: plain markdown documentation

All file I/O uses only the Python standard library (3.10+).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """Result of a single skill export/import operation.

    Attributes:
        source_skill_id: The original skill ID from the registry.
        target_format: The export format used.
        target_path: File path where the skill was written.
        success: Whether the operation succeeded.
        notes: Human-readable notes about the transfer.
    """

    source_skill_id: str
    target_format: str
    target_path: str
    success: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# SkillTransferEngine
# ---------------------------------------------------------------------------

class SkillTransferEngine:
    """Exports skills to and imports skills from various agent formats.

    Supports batch operations and format conversion between different
    agent ecosystems (Hermes, OpenClaw, JSON, Markdown).

    Parameters:
        registry: The skill registry to read from / write to.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Export: Hermes (SKILL.md with YAML frontmatter)
    # ------------------------------------------------------------------

    def export_to_hermes(
        self, skill_ids: list[str], output_dir: str
    ) -> list[TransferResult]:
        """Export skills as Hermes-compatible SKILL.md files.

        Each file contains YAML frontmatter (name, version, tags,
        metadata) followed by the markdown body with tier-2 core
        instructions.

        Parameters:
            skill_ids: Skills to export.
            output_dir: Directory to write SKILL.md files to.

        Returns:
            List of :class:`TransferResult`.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[TransferResult] = []

        for skill_id in skill_ids:
            skill = self._registry.get_skill(skill_id, tier=3)
            if skill is None:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="hermes",
                    target_path="",
                    success=False,
                    notes=f"Skill '{skill_id}' not found in registry.",
                ))
                continue

            safe_name = self._sanitize_filename(skill.name)
            file_path = output_path / f"{safe_name}.skill.md"

            try:
                content = self._skill_to_hermes_md(skill)
                file_path.write_text(content, encoding="utf-8")
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="hermes",
                    target_path=str(file_path),
                    success=True,
                    notes=f"Exported to {file_path.name}",
                ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="hermes",
                    target_path=str(file_path),
                    success=False,
                    notes=f"Export failed: {exc}",
                ))

        return results

    def _skill_to_hermes_md(self, skill: Skill) -> str:
        """Convert a skill to Hermes YAML frontmatter + markdown format."""
        # YAML frontmatter
        tags_yaml = json.dumps(skill.tags)
        resources_yaml = json.dumps(skill.tier3_resources)

        lines = [
            "---",
            f"name: {skill.name}",
            f"version: {skill.version}",
            f"lifecycle: {skill.lifecycle.value}",
            f"q_value: {skill.q_value}",
            f"success_rate: {skill.success_rate}",
            f"tags: {tags_yaml}",
            f"tier1_metadata: \"{skill.tier1_metadata}\"",
            f"tier3_resources: {resources_yaml}",
            "---",
            "",
            f"# {skill.name}",
            "",
        ]

        if skill.tier1_metadata:
            lines.append(f"> {skill.tier1_metadata}")
            lines.append("")

        if skill.tier2_core:
            lines.append("## Instructions")
            lines.append("")
            lines.append(skill.tier2_core)
            lines.append("")

        if skill.tier3_resources:
            lines.append("## Resources")
            lines.append("")
            for res in skill.tier3_resources:
                lines.append(f"- {res}")
            lines.append("")

        lines.append(f"*Generated by SkillForge on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export: OpenClaw (.learnings/ format)
    # ------------------------------------------------------------------

    def export_to_openclaw(
        self, skill_ids: list[str], output_dir: str
    ) -> list[TransferResult]:
        """Export skills in OpenClaw ``.learnings/`` format.

        Each skill is written as a structured text file with key=value
        headers and the instruction body indented.

        Parameters:
            skill_ids: Skills to export.
            output_dir: Directory for the ``.learnings/`` files.

        Returns:
            List of :class:`TransferResult`.
        """
        output_path = Path(output_dir) / ".learnings"
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[TransferResult] = []

        for skill_id in skill_ids:
            skill = self._registry.get_skill(skill_id, tier=3)
            if skill is None:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="openclaw",
                    target_path="",
                    success=False,
                    notes=f"Skill '{skill_id}' not found in registry.",
                ))
                continue

            safe_name = self._sanitize_filename(skill.name)
            file_path = output_path / f"{safe_name}.learning"

            try:
                content = self._skill_to_openclaw(skill)
                file_path.write_text(content, encoding="utf-8")
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="openclaw",
                    target_path=str(file_path),
                    success=True,
                    notes=f"Exported to .learnings/{file_path.name}",
                ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="openclaw",
                    target_path=str(file_path),
                    success=False,
                    notes=f"Export failed: {exc}",
                ))

        return results

    def _skill_to_openclaw(self, skill: Skill) -> str:
        """Convert a skill to OpenClaw .learnings format."""
        lines = [
            f"# {skill.name}",
            f"source: skillforge",
            f"id: {skill.id}",
            f"version: {skill.version}",
            f"confidence: {skill.q_value}",
            f"tags: {', '.join(skill.tags)}",
            "",
            "## Summary",
            skill.tier1_metadata,
            "",
        ]

        if skill.tier2_core:
            lines.append("## Details")
            lines.append("")
            # OpenClaw expects indented content
            for line in skill.tier2_core.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export: JSON
    # ------------------------------------------------------------------

    def export_to_json(
        self, skill_ids: list[str], output_path: str
    ) -> list[TransferResult]:
        """Export skills as a JSON array.

        The JSON file contains a list of skill objects with all fields.

        Parameters:
            skill_ids: Skills to export.
            output_path: Path to the JSON file.

        Returns:
            List of :class:`TransferResult`.
        """
        skills_data: list[dict[str, Any]] = []
        results: list[TransferResult] = []

        for skill_id in skill_ids:
            skill = self._registry.get_skill(skill_id, tier=3)
            if skill is None:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="json",
                    target_path=output_path,
                    success=False,
                    notes=f"Skill '{skill_id}' not found in registry.",
                ))
                continue

            skills_data.append(self._skill_to_dict(skill))
            results.append(TransferResult(
                source_skill_id=skill_id,
                target_format="json",
                target_path=output_path,
                success=True,
                notes=f"Added to JSON export",
            ))

        # Write the JSON file
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(skills_data, f, indent=2, default=str)
        except Exception as exc:
            for r in results:
                if r.success:
                    r.success = False
                    r.notes = f"JSON write failed: {exc}"

        return results

    def _skill_to_dict(self, skill: Skill) -> dict[str, Any]:
        """Convert a Skill to a JSON-serializable dict."""
        return {
            "id": skill.id,
            "name": skill.name,
            "version": skill.version,
            "lifecycle": skill.lifecycle.value,
            "tier1_metadata": skill.tier1_metadata,
            "tier2_core": skill.tier2_core,
            "tier3_resources": skill.tier3_resources,
            "q_value": skill.q_value,
            "success_rate": skill.success_rate,
            "usage_count": skill.usage_count,
            "tags": skill.tags,
            "created_at": skill.created_at.isoformat(),
            "updated_at": skill.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Export: Markdown
    # ------------------------------------------------------------------

    def export_to_markdown(
        self, skill_ids: list[str], output_dir: str
    ) -> list[TransferResult]:
        """Export skills as plain markdown documentation files.

        Each skill becomes a human-readable markdown file with metadata
        in a table and the full instructions rendered.

        Parameters:
            skill_ids: Skills to export.
            output_dir: Directory to write markdown files.

        Returns:
            List of :class:`TransferResult`.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[TransferResult] = []

        for skill_id in skill_ids:
            skill = self._registry.get_skill(skill_id, tier=3)
            if skill is None:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="markdown",
                    target_path="",
                    success=False,
                    notes=f"Skill '{skill_id}' not found in registry.",
                ))
                continue

            safe_name = self._sanitize_filename(skill.name)
            file_path = output_path / f"{safe_name}.md"

            try:
                content = self._skill_to_plain_md(skill)
                file_path.write_text(content, encoding="utf-8")
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="markdown",
                    target_path=str(file_path),
                    success=True,
                    notes=f"Exported to {file_path.name}",
                ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=skill_id,
                    target_format="markdown",
                    target_path=str(file_path),
                    success=False,
                    notes=f"Export failed: {exc}",
                ))

        return results

    def _skill_to_plain_md(self, skill: Skill) -> str:
        """Convert a skill to plain markdown documentation."""
        lines = [
            f"# {skill.name}",
            "",
            f"**ID:** `{skill.id}`",
            f"**Version:** {skill.version}",
            f"**Status:** {skill.lifecycle.value}",
            f"**Q-value:** {skill.q_value}",
            f"**Success rate:** {skill.success_rate:.1%}",
            f"**Usage count:** {skill.usage_count}",
            f"**Tags:** {', '.join(skill.tags) if skill.tags else 'none'}",
            "",
        ]

        if skill.tier1_metadata:
            lines.append("## Description")
            lines.append("")
            lines.append(skill.tier1_metadata)
            lines.append("")

        if skill.tier2_core:
            lines.append("## Instructions")
            lines.append("")
            lines.append(skill.tier2_core)
            lines.append("")

        if skill.tier3_resources:
            lines.append("## Resources")
            lines.append("")
            for res in skill.tier3_resources:
                lines.append(f"- {res}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Exported by SkillForge on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_from_directory(
        self,
        input_dir: str,
        format: str = "hermes",
    ) -> list[TransferResult]:
        """Import skills from files in a directory.

        Parameters:
            input_dir: Directory containing skill files.
            format: The format of the source files.
                One of ``'hermes'``, ``'json'``, ``'markdown'``.

        Returns:
            List of :class:`TransferResult` for each imported file.
        """
        input_path = Path(input_dir)

        if not input_path.is_dir():
            return [TransferResult(
                source_skill_id="",
                target_format=format,
                target_path=input_dir,
                success=False,
                notes=f"Input directory '{input_dir}' does not exist.",
            )]

        if format == "hermes":
            return self._import_hermes(input_path)
        elif format == "json":
            return self._import_json(input_path)
        elif format == "markdown":
            return self._import_markdown(input_path)
        else:
            return [TransferResult(
                source_skill_id="",
                target_format=format,
                target_path=input_dir,
                success=False,
                notes=f"Unknown format: '{format}'. Use 'hermes', 'json', or 'markdown'.",
            )]

    def _import_hermes(self, directory: Path) -> list[TransferResult]:
        """Import from Hermes SKILL.md files."""
        results: list[TransferResult] = []

        md_files = list(directory.glob("*.skill.md")) + list(directory.glob("*.md"))
        # Deduplicate
        md_files = list(set(md_files))

        for md_file in sorted(md_files):
            try:
                content = md_file.read_text(encoding="utf-8")
                skill_data = self._parse_hermes_md(content)
                if skill_data is None:
                    results.append(TransferResult(
                        source_skill_id=md_file.stem,
                        target_format="hermes",
                        target_path=str(md_file),
                        success=False,
                        notes=f"Could not parse frontmatter in {md_file.name}",
                    ))
                    continue

                skill = self._registry.register_skill(
                    name=skill_data.get("name", md_file.stem),
                    tier1_metadata=skill_data.get("tier1_metadata", ""),
                    tier2_core=skill_data.get("tier2_core", ""),
                    tier3_resources=skill_data.get("tier3_resources", []),
                    tags=skill_data.get("tags", []),
                )
                results.append(TransferResult(
                    source_skill_id=skill.id,
                    target_format="hermes",
                    target_path=str(md_file),
                    success=True,
                    notes=f"Imported as skill '{skill.id}'",
                ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=md_file.stem,
                    target_format="hermes",
                    target_path=str(md_file),
                    success=False,
                    notes=f"Import failed: {exc}",
                ))

        return results

    def _parse_hermes_md(self, content: str) -> dict[str, Any] | None:
        """Parse a Hermes SKILL.md file into a dict."""
        # Extract YAML frontmatter between --- markers
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return None

        frontmatter = match.group(1)
        body = content[match.end():]

        data: dict[str, Any] = {}

        # Parse simple YAML-like key-value pairs
        for line in frontmatter.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Parse typed values
            if value.startswith("[") or value.startswith('"['):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    data[key] = value
            elif value.replace(".", "").replace("-", "").isdigit():
                try:
                    data[key] = float(value) if "." in value else int(value)
                except ValueError:
                    data[key] = value
            else:
                data[key] = value.strip('"').strip("'")

        # Extract tier2_core from the body (after ## Instructions heading)
        body_match = re.search(
            r"##\s*Instructions\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL
        )
        if body_match:
            data["tier2_core"] = body_match.group(1).strip()

        return data

    def _import_json(self, directory: Path) -> list[TransferResult]:
        """Import from JSON files."""
        results: list[TransferResult] = []

        json_files = list(directory.glob("*.json"))

        for json_file in json_files:
            try:
                content = json_file.read_text(encoding="utf-8")
                skill_list = json.loads(content)
                if isinstance(skill_list, dict):
                    skill_list = [skill_list]
                if not isinstance(skill_list, list):
                    results.append(TransferResult(
                        source_skill_id=json_file.stem,
                        target_format="json",
                        target_path=str(json_file),
                        success=False,
                        notes=f"Expected a JSON array or object in {json_file.name}",
                    ))
                    continue

                for skill_data in skill_list:
                    skill = self._registry.register_skill(
                        name=skill_data.get("name", "imported-skill"),
                        tier1_metadata=skill_data.get("tier1_metadata", ""),
                        tier2_core=skill_data.get("tier2_core", ""),
                        tier3_resources=skill_data.get("tier3_resources", []),
                        tags=skill_data.get("tags", []),
                    )
                    results.append(TransferResult(
                        source_skill_id=skill.id,
                        target_format="json",
                        target_path=str(json_file),
                        success=True,
                        notes=f"Imported as skill '{skill.id}'",
                    ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=json_file.stem,
                    target_format="json",
                    target_path=str(json_file),
                    success=False,
                    notes=f"Import failed: {exc}",
                ))

        return results

    def _import_markdown(self, directory: Path) -> list[TransferResult]:
        """Import from plain markdown files."""
        results: list[TransferResult] = []

        md_files = list(directory.glob("*.md"))
        # Skip hermes-style files
        md_files = [f for f in md_files if not f.name.endswith(".skill.md")]

        for md_file in sorted(md_files):
            try:
                content = md_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Extract title from first heading
                name = md_file.stem
                for line in lines:
                    if line.startswith("# "):
                        name = line[2:].strip()
                        break

                # Extract description
                description = ""
                desc_match = re.search(
                    r"##\s*Description\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
                )
                if desc_match:
                    description = desc_match.group(1).strip()

                # Extract instructions
                tier2 = ""
                inst_match = re.search(
                    r"##\s*Instructions\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
                )
                if inst_match:
                    tier2 = inst_match.group(1).strip()

                skill = self._registry.register_skill(
                    name=name,
                    tier1_metadata=description,
                    tier2_core=tier2,
                    tags=["imported"],
                )
                results.append(TransferResult(
                    source_skill_id=skill.id,
                    target_format="markdown",
                    target_path=str(md_file),
                    success=True,
                    notes=f"Imported as skill '{skill.id}'",
                ))
            except Exception as exc:
                results.append(TransferResult(
                    source_skill_id=md_file.stem,
                    target_format="markdown",
                    target_path=str(md_file),
                    success=False,
                    notes=f"Import failed: {exc}",
                ))

        return results

    # ------------------------------------------------------------------
    # Batch transfer
    # ------------------------------------------------------------------

    def batch_transfer(
        self,
        source_format: str,
        target_format: str,
        skill_ids: list[str] | None = None,
        source_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[TransferResult]:
        """Transfer skills between formats in a single batch operation.

        If ``skill_ids`` is provided, those skills are exported from the
        registry to ``target_format``.  If ``source_dir`` is provided
        instead, files are imported from that directory into the registry.

        Parameters:
            source_format: Source format (``'hermes'``, ``'json'``, ``'markdown'``).
            target_format: Target format (``'hermes'``, ``'openclaw'``, ``'json'``, ``'markdown'``).
            skill_ids: Skill IDs to export (mutually exclusive with source_dir).
            source_dir: Directory to import from (mutually exclusive with skill_ids).
            output_dir: Output directory/file path.

        Returns:
            List of :class:`TransferResult`.
        """
        if source_dir and not skill_ids:
            # Import from source_dir, then export
            import_results = self.import_from_directory(source_dir, format=source_format)
            imported_ids = [
                r.source_skill_id for r in import_results if r.success
            ]

            if not imported_ids:
                return import_results

            # Now export in target format
            export_results = self._do_export(
                imported_ids, target_format, output_dir
            )
            return import_results + export_results

        elif skill_ids:
            # Direct export
            return self._do_export(skill_ids, target_format, output_dir)

        else:
            # Export all skills from registry
            all_skills = self._registry.list_skills()
            all_ids = [s.id for s in all_skills]
            return self._do_export(all_ids, target_format, output_dir)

    def _do_export(
        self,
        skill_ids: list[str],
        target_format: str,
        output_dir: str | None,
    ) -> list[TransferResult]:
        """Dispatch to the correct export method based on target format.

        Parameters:
            skill_ids: Skills to export.
            target_format: Target format name.
            output_dir: Output directory.

        Returns:
            List of transfer results.
        """
        if output_dir is None:
            output_dir = str(Path.cwd() / "skillforge_export")

        if target_format == "hermes":
            return self.export_to_hermes(skill_ids, output_dir)
        elif target_format == "openclaw":
            return self.export_to_openclaw(skill_ids, output_dir)
        elif target_format == "json":
            output_path = Path(output_dir) / "skills_export.json"
            return self.export_to_json(skill_ids, str(output_path))
        elif target_format == "markdown":
            return self.export_to_markdown(skill_ids, output_dir)
        else:
            return [TransferResult(
                source_skill_id=sid,
                target_format=target_format,
                target_path="",
                success=False,
                notes=f"Unknown target format: '{target_format}'",
            ) for sid in skill_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert a skill name to a safe filename."""
        # Remove special characters, replace spaces with hyphens
        safe = re.sub(r"[^\w\s-]", "", name.lower())
        safe = re.sub(r"[-\s]+", "-", safe).strip("-")
        return safe or "unnamed"
