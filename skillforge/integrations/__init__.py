"""SkillForge Integrations — adapters and platform bridges."""

from skillforge.integrations.adapters import (
    IntegrationAdapter,
    AdapterRegistry,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    EventPayload,
    EventType,
    GitHubAdapter,
    SlackAdapter,
    WebhookAdapter,
    FileAdapter,
)

__all__ = [
    "IntegrationAdapter",
    "AdapterRegistry",
    "AdapterConfig",
    "AdapterResult",
    "AdapterStatus",
    "EventPayload",
    "EventType",
    "GitHubAdapter",
    "SlackAdapter",
    "WebhookAdapter",
    "FileAdapter",
]
