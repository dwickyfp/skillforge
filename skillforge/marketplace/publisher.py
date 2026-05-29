"""Skill publisher — package and publish skills to the marketplace.

Wraps :class:`MarketplaceRegistry` to provide a higher-level publish
workflow that validates skills before submission.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .registry import MarketplaceEntry, MarketplaceRegistry, MarketplaceStatus


# Semantic version regex
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-.]+))?(?:\+(?P<build>[0-9A-Za-z\-.]+))?$"
)


@dataclass
class PublishResult:
    """Result of a publish operation.

    Attributes:
        success: Whether the publish succeeded.
        entry: The published marketplace entry (on success).
        errors: List of validation error messages (on failure).
        warnings: Non-fatal warnings.
        published_at: Timestamp of publication.
    """

    success: bool
    entry: MarketplaceEntry | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SkillPublisher:
    """High-level publisher that validates and publishes skills.

    Parameters
    ----------
    marketplace : MarketplaceRegistry
        The marketplace registry to publish to.
    """

    def __init__(self, marketplace: MarketplaceRegistry) -> None:
        self._marketplace = marketplace

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_skill(
        skill_name: str,
        publisher: str,
        version: str = "1.0.0",
        tier1_metadata: str = "",
        tier2_core: str = "",
        tags: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Validate a skill before publishing.

        Parameters
        ----------
        skill_name : str
            Skill name (required, 2–100 chars).
        publisher : str
            Publisher name (required, 2–50 chars).
        version : str
            Semantic version string.
        tier1_metadata : str
            Short routing summary.
        tier2_core : str
            Core prompt / instructions.
        tags : list[str] | None
            Tags for the listing.

        Returns
        -------
        tuple[list[str], list[str]]
            (errors, warnings) — errors are fatal, warnings are not.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Name validation
        if not skill_name or len(skill_name.strip()) < 2:
            errors.append("Skill name must be at least 2 characters")
        if len(skill_name) > 100:
            errors.append("Skill name must be at most 100 characters")

        # Publisher validation
        if not publisher or len(publisher.strip()) < 2:
            errors.append("Publisher name must be at least 2 characters")
        if len(publisher) > 50:
            errors.append("Publisher name must be at most 50 characters")

        # Version validation
        if not _SEMVER_RE.match(version):
            errors.append(
                f"Invalid semantic version '{version}'. "
                "Expected format: MAJOR.MINOR.PATCH"
            )

        # Content validation
        if not tier1_metadata:
            warnings.append("tier1_metadata is empty — routing quality may suffer")
        elif len(tier1_metadata) > 500:
            warnings.append(
                f"tier1_metadata is {len(tier1_metadata)} chars — "
                "recommend under 500 for routing efficiency"
            )

        if not tier2_core:
            warnings.append("tier2_core is empty — skill has no instructions")

        # Tag validation
        if not tags:
            warnings.append("No tags provided — discoverability may be reduced")
        elif len(tags) > 20:
            warnings.append(f"Too many tags ({len(tags)}) — recommend at most 20")

        return errors, warnings

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_skill(
        self,
        skill_name: str,
        publisher: str,
        description: str = "",
        version: str = "1.0.0",
        tier1_metadata: str = "",
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        license: str = "MIT",
        listing_id: str | None = None,
    ) -> PublishResult:
        """Validate and publish a skill to the marketplace.

        Parameters
        ----------
        skill_name : str
            Human-readable skill name.
        publisher : str
            Author / publisher name.
        description : str
            Short description for the listing.
        version : str
            Semantic version string.
        tier1_metadata : str
            Routing metadata (~30 tokens).
        tier2_core : str
            Core prompt / instructions.
        tier3_resources : list[str] | None
            Auxiliary resource paths.
        tags : list[str] | None
            Searchable tags.
        license : str
            License identifier.
        listing_id : str | None
            Explicit listing ID.  A UUID is generated when *None*.

        Returns
        -------
        PublishResult
            Result containing the entry on success, or errors on failure.
        """
        errors, warnings = self.validate_skill(
            skill_name=skill_name,
            publisher=publisher,
            version=version,
            tier1_metadata=tier1_metadata,
            tier2_core=tier2_core,
            tags=tags,
        )

        if errors:
            return PublishResult(success=False, errors=errors, warnings=warnings)

        try:
            entry = self._marketplace.publish(
                skill_name=skill_name,
                publisher=publisher,
                description=description,
                version=version,
                tier1_metadata=tier1_metadata,
                tier2_core=tier2_core,
                tier3_resources=tier3_resources,
                tags=tags,
                license=license,
                listing_id=listing_id,
                status=MarketplaceStatus.PUBLISHED,
            )
            return PublishResult(success=True, entry=entry, warnings=warnings)
        except ValueError as exc:
            return PublishResult(
                success=False,
                errors=[str(exc)],
                warnings=warnings,
            )

    def update_published_skill(
        self,
        listing_id: str,
        updates: dict[str, Any],
    ) -> PublishResult:
        """Update an existing published listing.

        Parameters
        ----------
        listing_id : str
            Listing to update.
        updates : dict
            Fields to update.

        Returns
        -------
        PublishResult
            Result of the update operation.
        """
        # If version is being updated, validate it
        if "version" in updates:
            if not _SEMVER_RE.match(updates["version"]):
                return PublishResult(
                    success=False,
                    errors=[
                        f"Invalid semantic version '{updates['version']}'. "
                        "Expected format: MAJOR.MINOR.PATCH"
                    ],
                )

        try:
            entry = self._marketplace.update_listing(listing_id, updates)
            if entry is None:
                return PublishResult(
                    success=False,
                    errors=[f"Listing '{listing_id}' not found"],
                )
            return PublishResult(success=True, entry=entry)
        except Exception as exc:
            return PublishResult(success=False, errors=[str(exc)])

    def deprecate_skill(self, listing_id: str) -> bool:
        """Deprecate a published skill.

        Parameters
        ----------
        listing_id : str
            Listing to deprecate.

        Returns
        -------
        bool
            True if deprecation succeeded.
        """
        entry = self._marketplace.deprecate_listing(listing_id)
        return entry is not None

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------

    @staticmethod
    def package_skill(
        skill_name: str,
        tier1_metadata: str,
        tier2_core: str,
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Package a skill into a portable dictionary format.

        Parameters
        ----------
        skill_name : str
            Skill name.
        tier1_metadata : str
            Routing metadata.
        tier2_core : str
            Core instructions.
        tier3_resources : list[str] | None
            Resource list.
        tags : list[str] | None
            Tags.
        version : str
            Semantic version.

        Returns
        -------
        dict[str, Any]
            Portable skill package dictionary.
        """
        return {
            "format": "skillforge-package",
            "format_version": 1,
            "skill_name": skill_name,
            "version": version,
            "tier1_metadata": tier1_metadata,
            "tier2_core": tier2_core,
            "tier3_resources": tier3_resources or [],
            "tags": tags or [],
        }

    @staticmethod
    def serialize_package(package: dict[str, Any]) -> str:
        """Serialize a skill package to a JSON string.

        Parameters
        ----------
        package : dict[str, Any]
            Skill package dictionary.

        Returns
        -------
        str
            JSON string representation.
        """
        return json.dumps(package, indent=2, ensure_ascii=False)

    @staticmethod
    def deserialize_package(data: str) -> dict[str, Any]:
        """Deserialize a JSON string into a skill package.

        Parameters
        ----------
        data : str
            JSON string.

        Returns
        -------
        dict[str, Any]
            Skill package dictionary.

        Raises
        ------
        ValueError
            If the data is not a valid skill package.
        """
        package = json.loads(data)
        if not isinstance(package, dict):
            raise ValueError("Package must be a JSON object")
        if package.get("format") != "skillforge-package":
            raise ValueError(
                f"Unknown package format: {package.get('format')!r}"
            )
        return package
