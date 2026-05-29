"""Skill installer — install marketplace skills into a local registry.

Handles fetching marketplace listings, resolving dependencies, and
importing skills into a :class:`skillforge.core.registry.SkillRegistry`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .registry import MarketplaceEntry, MarketplaceRegistry


@dataclass
class InstallResult:
    """Result of an installation operation.

    Attributes:
        success: Whether the install succeeded.
        listing_id: Marketplace listing that was installed.
        registry_id: ID assigned in the local skill registry.
        installed_version: Version that was installed.
        errors: List of error messages if the install failed.
        installed_at: Timestamp of the install.
    """

    success: bool
    listing_id: str = ""
    registry_id: str = ""
    installed_version: str = ""
    errors: list[str] = field(default_factory=list)
    installed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SkillInstaller:
    """Install skills from the marketplace into a local registry.

    Parameters
    ----------
    marketplace : MarketplaceRegistry
        The marketplace to install from.
    """

    def __init__(self, marketplace: MarketplaceRegistry) -> None:
        self._marketplace = marketplace

    def install(
        self,
        listing_id: str,
        registry: Any | None = None,
    ) -> InstallResult:
        """Install a marketplace skill into a local registry.

        Parameters
        ----------
        listing_id : str
            Marketplace listing ID to install.
        registry : Any | None
            Optional local :class:`SkillRegistry` instance.  If *None* the
            skill is only recorded as installed (no local registry import).

        Returns
        -------
        InstallResult
            Installation result.
        """
        # Fetch listing
        entry = self._marketplace.get_listing(listing_id)
        if entry is None:
            return InstallResult(
                success=False,
                listing_id=listing_id,
                errors=[f"Listing '{listing_id}' not found"],
            )

        from .registry import MarketplaceStatus

        # Allow installing published or unlisted (but not pending/deprecated)
        if entry.status not in (MarketplaceStatus.PUBLISHED, MarketplaceStatus.UNLISTED):
            return InstallResult(
                success=False,
                listing_id=listing_id,
                errors=[
                    f"Listing status is '{entry.status.value}' — "
                    "only published or unlisted skills can be installed"
                ],
            )

        registry_id = ""
        try:
            if registry is not None:
                # Import into local registry
                combined_tags = list(entry.tags) + [
                    f"marketplace:{listing_id}",
                    f"publisher:{entry.publisher}",
                    f"version:{entry.version}",
                ]
                skill = registry.register_skill(
                    name=entry.skill_name,
                    tier1_metadata=entry.tier1_metadata or entry.description,
                    tier2_core=entry.tier2_core,
                    tier3_resources=entry.tier3_resources,
                    tags=combined_tags,
                )
                registry_id = skill.id
            else:
                registry_id = str(uuid.uuid4())

            # Record the install
            self._marketplace.record_install(
                listing_id=listing_id,
                installed_version=entry.version,
                target_registry_id=registry_id,
            )

            return InstallResult(
                success=True,
                listing_id=listing_id,
                registry_id=registry_id,
                installed_version=entry.version,
            )

        except Exception as exc:
            return InstallResult(
                success=False,
                listing_id=listing_id,
                errors=[f"Installation failed: {exc}"],
            )

    def get_installed_version(self, listing_id: str) -> str | None:
        """Get the latest installed version for a listing.

        Parameters
        ----------
        listing_id : str
            Marketplace listing ID.

        Returns
        -------
        str | None
            The latest installed version string, or *None* if never installed.
        """
        history = self._marketplace.get_install_history(listing_id, limit=1)
        if history:
            return history[0].installed_version
        return None

    def check_update(self, listing_id: str) -> tuple[str | None, str | None]:
        """Check if an update is available for an installed skill.

        Parameters
        ----------
        listing_id : str
            Marketplace listing ID.

        Returns
        -------
        tuple[str | None, str | None]
            (installed_version, latest_version), or (None, None) if
            not installed or listing not found.
        """
        installed = self.get_installed_version(listing_id)
        if installed is None:
            return None, None

        entry = self._marketplace.get_listing(listing_id)
        if entry is None:
            return None, None

        return installed, entry.version

    def uninstall(
        self,
        listing_id: str,
        registry: Any | None = None,
    ) -> bool:
        """Uninstall a marketplace skill from the local registry.

        Parameters
        ----------
        listing_id : str
            Marketplace listing ID to uninstall.
        registry : Any | None
            Local registry to remove the skill from.

        Returns
        -------
        bool
            True if uninstall was successful.
        """
        if registry is None:
            return True

        try:
            history = self._marketplace.get_install_history(listing_id, limit=1)
            if not history:
                return False

            target_id = history[0].target_registry_id
            if target_id:
                skill = registry.get_skill(target_id, tier=3)
                if skill is not None:
                    registry.update_skill(
                        target_id, {"lifecycle": "archived"}
                    )
                    return True
            return False
        except Exception:
            return False
