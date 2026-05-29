"""Tests for Phase 6a modules: Marketplace, Observability, Async Support, Versioning.

Uses pytest with tmp_path for DB isolation.  Tests cover:
- Marketplace: publish, search, rate, install, deprecate
- Observability: tracing spans, metrics collection, structured logging
- Async Support: async registry, tracker, loader operations
- Versioning: semantic version parsing, bumping, history, rollback, diff
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import pytest

from skillforge.core.registry import Skill, SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.core.loader import ProgressiveLoader

from skillforge.marketplace.registry import (
    MarketplaceRegistry,
    MarketplaceEntry,
    MarketplaceStatus,
    InstallRecord,
)
from skillforge.marketplace.publisher import SkillPublisher, PublishResult
from skillforge.marketplace.installer import SkillInstaller, InstallResult

from skillforge.observability.tracer import SkillTracer, Span, SpanStatus, TraceContext
from skillforge.observability.metrics import MetricsCollector, MetricPoint, MetricSummary
from skillforge.observability.logger import StructuredLogger, LogEntry, LogLevel

from skillforge.async_support.async_registry import AsyncSkillRegistry
from skillforge.async_support.async_tracker import AsyncQValueTracker
from skillforge.async_support.async_loader import AsyncProgressiveLoader

from skillforge.versioning.version_manager import (
    VersionManager,
    SemanticVersion,
    VersionRecord,
    VersionDiff,
    ChangeType,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    """Create a temp directory for DB files."""
    d = tmp_path / "db"
    d.mkdir()
    return d


@pytest.fixture
def registry(db_dir: Path) -> SkillRegistry:
    """Fresh registry backed by a temp DB."""
    reg = SkillRegistry(db_path=db_dir / "skills.db")
    yield reg  # type: ignore[misc]
    reg.close()


@pytest.fixture
def tracker(db_dir: Path) -> QValueTracker:
    """Fresh tracker backed by a temp DB."""
    tr = QValueTracker(db_path=db_dir / "tracker.db")
    yield tr  # type: ignore[misc]
    tr.close()


@pytest.fixture
def marketplace(db_dir: Path) -> MarketplaceRegistry:
    """Fresh marketplace backed by a temp DB."""
    mp = MarketplaceRegistry(db_path=db_dir / "marketplace.db")
    yield mp  # type: ignore[misc]
    mp.close()


@pytest.fixture
def publisher(marketplace: MarketplaceRegistry) -> SkillPublisher:
    """Publisher wired to the test marketplace."""
    return SkillPublisher(marketplace)


@pytest.fixture
def installer(marketplace: MarketplaceRegistry) -> SkillInstaller:
    """Installer wired to the test marketplace."""
    return SkillInstaller(marketplace)


@pytest.fixture
def tracer(db_dir: Path) -> SkillTracer:
    """Fresh tracer backed by a temp DB."""
    t = SkillTracer(db_path=db_dir / "traces.db")
    yield t  # type: ignore[misc]
    t.close()


@pytest.fixture
def metrics(db_dir: Path) -> MetricsCollector:
    """Fresh metrics collector backed by a temp DB."""
    m = MetricsCollector(db_path=db_dir / "metrics.db")
    yield m  # type: ignore[misc]
    m.close()


@pytest.fixture
def logger(db_dir: Path) -> StructuredLogger:
    """Fresh structured logger backed by a temp DB."""
    l = StructuredLogger(db_path=db_dir / "logs.db")
    yield l  # type: ignore[misc]
    l.close()


@pytest.fixture
def version_mgr(db_dir: Path) -> VersionManager:
    """Fresh version manager backed by a temp DB."""
    vm = VersionManager(db_path=db_dir / "versions.db")
    yield vm  # type: ignore[misc]
    vm.close()


# =======================================================================
# TestMarketplaceRegistry
# =======================================================================


class TestMarketplaceRegistry:
    """Tests for the MarketplaceRegistry module."""

    def test_publish_and_get(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Publishing a skill and retrieving it should work."""
        entry = marketplace.publish(
            skill_name="web-scraper",
            publisher="alice",
            description="Scrapes web content",
            tier1_metadata="Web scraping skill",
            tier2_core="1. Fetch URL\n2. Parse HTML",
            tags=["web", "scraping"],
        )
        assert isinstance(entry, MarketplaceEntry)
        assert entry.skill_name == "web-scraper"
        assert entry.status == MarketplaceStatus.PUBLISHED
        assert entry.version == "1.0.0"

        fetched = marketplace.get_listing(entry.id)
        assert fetched is not None
        assert fetched.skill_name == "web-scraper"

    def test_publish_with_custom_id(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Publishing with a custom ID should use it."""
        entry = marketplace.publish(
            skill_name="api-fetcher",
            publisher="bob",
            listing_id="custom-listing-1",
        )
        assert entry.id == "custom-listing-1"

    def test_publish_duplicate_id_raises(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Publishing with an existing ID should raise ValueError."""
        marketplace.publish(
            skill_name="a", publisher="b", listing_id="dup"
        )
        with pytest.raises(ValueError, match="already exists"):
            marketplace.publish(
                skill_name="c", publisher="d", listing_id="dup"
            )

    def test_get_nonexistent_listing(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Getting a nonexistent listing should return None."""
        assert marketplace.get_listing("nonexistent") is None

    def test_update_listing(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Updating a listing should apply changes."""
        entry = marketplace.publish(skill_name="skill-x", publisher="alice")
        updated = marketplace.update_listing(
            entry.id, {"description": "Updated description", "version": "1.1.0"}
        )
        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.version == "1.1.0"

    def test_update_nonexistent(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Updating a nonexistent listing should return None."""
        assert marketplace.update_listing("nope", {"description": "x"}) is None

    def test_list_listings_by_status(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Listing by status should filter correctly."""
        marketplace.publish(
            skill_name="a", publisher="p",
            status=MarketplaceStatus.PUBLISHED,
        )
        marketplace.publish(
            skill_name="b", publisher="p",
            status=MarketplaceStatus.DEPRECATED,
        )

        published = marketplace.list_listings(status=MarketplaceStatus.PUBLISHED)
        assert len(published) == 1
        assert published[0].skill_name == "a"

    def test_list_listings_by_publisher(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Listing by publisher should filter correctly."""
        marketplace.publish(skill_name="a", publisher="alice")
        marketplace.publish(skill_name="b", publisher="bob")

        alice_listings = marketplace.list_listings(publisher="alice")
        assert len(alice_listings) == 1
        assert alice_listings[0].publisher == "alice"

    def test_list_listings_by_tags(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Listing by tags should match skills with overlapping tags."""
        marketplace.publish(skill_name="a", publisher="p", tags=["web", "api"])
        marketplace.publish(skill_name="b", publisher="p", tags=["data", "ml"])

        results = marketplace.list_listings(tags=["web"])
        assert len(results) == 1
        assert results[0].skill_name == "a"

    def test_search(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Search should find matching skills."""
        marketplace.publish(
            skill_name="web-scraper",
            publisher="p",
            description="Scrapes websites for data",
        )
        marketplace.publish(
            skill_name="email-sender",
            publisher="p",
            description="Sends emails",
        )

        results = marketplace.search("web")
        assert len(results) >= 1
        assert any("web" in r.skill_name.lower() for r in results)

    def test_rate_listing(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Rating should update the average and count."""
        entry = marketplace.publish(skill_name="a", publisher="p")

        marketplace.rate_listing(entry.id, 4.0)
        marketplace.rate_listing(entry.id, 5.0)

        updated = marketplace.get_listing(entry.id)
        assert updated is not None
        assert updated.rating == pytest.approx(4.5, abs=0.01)
        assert updated.rating_count == 2

    def test_rate_listing_clamped(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Ratings should be clamped to [0, 5]."""
        entry = marketplace.publish(skill_name="a", publisher="p")
        marketplace.rate_listing(entry.id, 10.0)
        marketplace.rate_listing(entry.id, -5.0)

        updated = marketplace.get_listing(entry.id)
        assert updated is not None
        # First rating: (0 + 5) / 1 = 5.0 (clamped from 10)
        # Second rating: (5 + 0) / 2 = 2.5 (clamped from -5)
        assert updated.rating == pytest.approx(2.5, abs=0.01)

    def test_record_install(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Recording an install should bump the count."""
        entry = marketplace.publish(skill_name="a", publisher="p")

        record = marketplace.record_install(
            entry.id, installed_version="1.0.0", target_registry_id="reg-1"
        )
        assert isinstance(record, InstallRecord)
        assert record.listing_id == entry.id

        count = marketplace.get_install_count(entry.id)
        assert count == 1

    def test_install_history(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Install history should track multiple installs."""
        entry = marketplace.publish(skill_name="a", publisher="p")

        marketplace.record_install(entry.id, installed_version="1.0.0")
        marketplace.record_install(entry.id, installed_version="1.1.0")

        history = marketplace.get_install_history(entry.id)
        assert len(history) == 2
        # Most recent first
        assert history[0].installed_version == "1.1.0"

    def test_deprecate_listing(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Deprecating a listing should change its status."""
        entry = marketplace.publish(skill_name="a", publisher="p")
        deprecated = marketplace.deprecate_listing(entry.id)
        assert deprecated is not None
        assert deprecated.status == MarketplaceStatus.DEPRECATED

    def test_count(
        self, marketplace: MarketplaceRegistry
    ) -> None:
        """Count should return accurate totals."""
        assert marketplace.count() == 0

        marketplace.publish(skill_name="a", publisher="p")
        marketplace.publish(skill_name="b", publisher="p")

        assert marketplace.count() == 2
        assert marketplace.count(status=MarketplaceStatus.PUBLISHED) == 2
        assert marketplace.count(status=MarketplaceStatus.DEPRECATED) == 0


# =======================================================================
# TestSkillPublisher
# =======================================================================


class TestSkillPublisher:
    """Tests for the SkillPublisher module."""

    def test_publish_valid_skill(
        self, publisher: SkillPublisher
    ) -> None:
        """Publishing a valid skill should succeed."""
        result = publisher.publish_skill(
            skill_name="web-scraper",
            publisher="alice",
            description="Scrapes web content",
            version="1.0.0",
            tier1_metadata="Web scraping",
            tier2_core="Instructions here",
            tags=["web"],
        )
        assert isinstance(result, PublishResult)
        assert result.success is True
        assert result.entry is not None
        assert len(result.errors) == 0

    def test_publish_invalid_name(
        self, publisher: SkillPublisher
    ) -> None:
        """Publishing with an invalid name should fail."""
        result = publisher.publish_skill(
            skill_name="x",
            publisher="alice",
        )
        assert result.success is False
        assert len(result.errors) >= 1
        assert any("name" in e.lower() for e in result.errors)

    def test_publish_invalid_version(
        self, publisher: SkillPublisher
    ) -> None:
        """Publishing with an invalid version should fail."""
        result = publisher.publish_skill(
            skill_name="web-scraper",
            publisher="alice",
            version="not-a-version",
        )
        assert result.success is False
        assert any("version" in e.lower() for e in result.errors)

    def test_publish_with_warnings(
        self, publisher: SkillPublisher
    ) -> None:
        """Publishing without tags should produce warnings."""
        result = publisher.publish_skill(
            skill_name="web-scraper",
            publisher="alice",
            version="1.0.0",
        )
        assert result.success is True
        assert len(result.warnings) >= 1
        assert any("tag" in w.lower() for w in result.warnings)

    def test_validate_skill_static(
        self,
    ) -> None:
        """Static validation should work independently."""
        errors, warnings = SkillPublisher.validate_skill(
            skill_name="valid-skill",
            publisher="alice",
            version="1.2.3",
            tier1_metadata="Good metadata",
            tier2_core="Instructions",
            tags=["a", "b"],
        )
        assert len(errors) == 0

    def test_validate_skill_errors(
        self,
    ) -> None:
        """Static validation should catch errors."""
        errors, warnings = SkillPublisher.validate_skill(
            skill_name="",
            publisher="",
            version="bad",
        )
        assert len(errors) >= 3  # name, publisher, version

    def test_deprecate_skill(
        self, publisher: SkillPublisher
    ) -> None:
        """Deprecating a published skill should work."""
        result = publisher.publish_skill(
            skill_name="web-scraper", publisher="alice"
        )
        assert result.entry is not None

        success = publisher.deprecate_skill(result.entry.id)
        assert success is True

    def test_package_and_serialize(
        self,
    ) -> None:
        """Packaging and serialization should round-trip."""
        package = SkillPublisher.package_skill(
            skill_name="test-skill",
            tier1_metadata="Test",
            tier2_core="Instructions",
            tags=["test"],
        )
        assert package["format"] == "skillforge-package"

        serialized = SkillPublisher.serialize_package(package)
        deserialized = SkillPublisher.deserialize_package(serialized)
        assert deserialized["skill_name"] == "test-skill"

    def test_deserialize_invalid_package(
        self,
    ) -> None:
        """Deserializing an invalid package should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown package format"):
            SkillPublisher.deserialize_package('{"format": "unknown"}')


# =======================================================================
# TestSkillInstaller
# =======================================================================


class TestSkillInstaller:
    """Tests for the SkillInstaller module."""

    def test_install_published_skill(
        self,
        installer: SkillInstaller,
        marketplace: MarketplaceRegistry,
    ) -> None:
        """Installing a published skill should succeed."""
        entry = marketplace.publish(
            skill_name="web-scraper",
            publisher="alice",
            status=MarketplaceStatus.PUBLISHED,
        )

        result = installer.install(entry.id)
        assert isinstance(result, InstallResult)
        assert result.success is True
        assert result.listing_id == entry.id

    def test_install_with_registry(
        self,
        installer: SkillInstaller,
        marketplace: MarketplaceRegistry,
        registry: SkillRegistry,
    ) -> None:
        """Installing into a local registry should work."""
        entry = marketplace.publish(
            skill_name="web-scraper",
            publisher="alice",
            tier1_metadata="Web scraping",
            tier2_core="Instructions",
            status=MarketplaceStatus.PUBLISHED,
        )

        result = installer.install(entry.id, registry=registry)
        assert result.success is True
        assert result.registry_id != ""

        # Check the skill was imported
        skill = registry.get_skill(result.registry_id, tier=1)
        assert skill is not None
        assert skill.name == "web-scraper"

    def test_install_nonexistent(
        self, installer: SkillInstaller
    ) -> None:
        """Installing a nonexistent listing should fail."""
        result = installer.install("nonexistent")
        assert result.success is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_install_deprecated(
        self,
        installer: SkillInstaller,
        marketplace: MarketplaceRegistry,
    ) -> None:
        """Installing a deprecated skill should fail."""
        entry = marketplace.publish(
            skill_name="old-skill",
            publisher="alice",
            status=MarketplaceStatus.DEPRECATED,
        )
        result = installer.install(entry.id)
        assert result.success is False
        assert any("deprecated" in e.lower() for e in result.errors)

    def test_check_update(
        self,
        installer: SkillInstaller,
        marketplace: MarketplaceRegistry,
    ) -> None:
        """Check update should compare versions."""
        entry = marketplace.publish(
            skill_name="skill-a",
            publisher="p",
            version="1.0.0",
            status=MarketplaceStatus.PUBLISHED,
        )

        # Install first
        installer.install(entry.id)

        # Update the listing
        marketplace.update_listing(entry.id, {"version": "1.1.0"})

        installed, latest = installer.check_update(entry.id)
        assert installed == "1.0.0"
        assert latest == "1.1.0"


# =======================================================================
# TestSkillTracer
# =======================================================================


class TestSkillTracer:
    """Tests for the SkillTracer module."""

    def test_new_trace(
        self, tracer: SkillTracer
    ) -> None:
        """Creating a new trace should return a context."""
        ctx = tracer.new_trace()
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id != ""

    def test_start_and_end_span(
        self, tracer: SkillTracer
    ) -> None:
        """Starting and ending a span should persist it."""
        span = tracer.start_span(
            operation="load_skill",
            skill_id="skill-1",
        )
        tracer.end_span(span, SpanStatus.OK)

        traces = tracer.get_trace(span.trace_id)
        assert len(traces) == 1
        assert traces[0].operation == "load_skill"
        assert traces[0].status == SpanStatus.OK

    def test_context_manager_span(
        self, tracer: SkillTracer
    ) -> None:
        """Context manager should auto-finish spans."""
        span: Span
        with tracer.trace_span(
            operation="query", skill_id="skill-1"
        ) as span:
            span.set_attribute("query_text", "hello")
            assert span.span_id != ""

        # Span should be persisted
        traces = tracer.get_trace(span.trace_id)
        assert len(traces) == 1
        assert traces[0].attributes["query_text"] == "hello"

    def test_context_manager_error_span(
        self, tracer: SkillTracer
    ) -> None:
        """Context manager should mark spans as ERROR on exception."""
        caught = False
        error_span: Span | None = None
        try:
            with tracer.trace_span(operation="failing_op") as span:
                error_span = span
                raise ValueError("boom")
        except ValueError:
            caught = True

        assert caught
        assert error_span is not None
        traces = tracer.get_trace(error_span.trace_id)
        assert len(traces) == 1
        assert traces[0].status == SpanStatus.ERROR

    def test_nested_spans(
        self, tracer: SkillTracer
    ) -> None:
        """Nested spans should maintain parent-child relationships."""
        ctx = tracer.new_trace()

        parent = tracer.start_span(
            operation="parent_op", trace_id=ctx.trace_id
        )
        child = tracer.start_span(
            operation="child_op",
            trace_id=ctx.trace_id,
            parent_span_id=parent.span_id,
        )

        tracer.end_span(child, SpanStatus.OK)
        tracer.end_span(parent, SpanStatus.OK)

        traces = tracer.get_trace(ctx.trace_id)
        assert len(traces) == 2
        # Child should reference parent
        child_span = [s for s in traces if s.operation == "child_op"][0]
        assert child_span.parent_span_id == parent.span_id

    def test_get_skill_spans(
        self, tracer: SkillTracer
    ) -> None:
        """Querying by skill_id should return matching spans."""
        for _ in range(3):
            with tracer.trace_span(operation="op", skill_id="skill-a"):
                pass
        with tracer.trace_span(operation="op", skill_id="skill-b"):
            pass

        spans = tracer.get_skill_spans("skill-a")
        assert len(spans) == 3

    def test_get_slow_spans(
        self, tracer: SkillTracer
    ) -> None:
        """Slow span queries should return spans above threshold."""
        # Create a span with artificial duration by persisting directly
        span = tracer.start_span(operation="slow_op")
        span.finish(SpanStatus.OK)
        # Override the duration after finish to simulate a slow span
        span.duration_ms = 5000.0
        tracer._persist_span(span)

        fast = tracer.start_span(operation="fast_op")
        fast.finish(SpanStatus.OK)
        fast.duration_ms = 10.0
        tracer._persist_span(fast)

        slow = tracer.get_slow_spans(threshold_ms=1000.0)
        assert len(slow) >= 1
        assert slow[0].duration_ms >= 1000.0

    def test_get_error_spans(
        self, tracer: SkillTracer
    ) -> None:
        """Error span queries should return error spans."""
        with tracer.trace_span(operation="ok_op"):
            pass

        err_span: Span | None = None
        try:
            with tracer.trace_span(operation="err_op") as span:
                err_span = span
                raise ValueError("test")
        except ValueError:
            pass

        errors = tracer.get_error_spans()
        assert len(errors) >= 1
        assert errors[0].status == SpanStatus.ERROR

    def test_get_trace_stats(
        self, tracer: SkillTracer
    ) -> None:
        """Trace stats should return aggregate information."""
        for i in range(5):
            with tracer.trace_span(operation=f"op-{i}"):
                pass

        stats = tracer.get_trace_stats()
        assert stats["total_spans"] >= 5
        assert stats["unique_traces"] >= 5
        assert "error_rate" in stats

    def test_span_events(
        self, tracer: SkillTracer
    ) -> None:
        """Span events should be recorded and retrievable."""
        span = tracer.start_span(operation="op")
        span.add_event("checkpoint_1", {"data": "value"})
        span.add_event("checkpoint_2")
        tracer.end_span(span)

        traces = tracer.get_trace(span.trace_id)
        assert len(traces[0].events) == 2
        assert traces[0].events[0]["name"] == "checkpoint_1"


# =======================================================================
# TestMetricsCollector
# =======================================================================


class TestMetricsCollector:
    """Tests for the MetricsCollector module."""

    def test_record_and_retrieve(
        self, metrics: MetricsCollector
    ) -> None:
        """Recording and retrieving a metric point should work."""
        point = metrics.record(
            name="skill.latency_ms",
            value=150.0,
            labels={"skill_id": "skill-1"},
        )
        assert isinstance(point, MetricPoint)
        assert point.value == 150.0

        points = metrics.get_metric_points("skill.latency_ms")
        assert len(points) == 1
        assert points[0].value == 150.0

    def test_record_counter(
        self, metrics: MetricsCollector
    ) -> None:
        """Counter recording should work."""
        metrics.record_counter("skill.invocations", increment=1.0)
        metrics.record_counter("skill.invocations", increment=1.0)

        points = metrics.get_metric_points("skill.invocations")
        assert len(points) == 2

    def test_record_gauge(
        self, metrics: MetricsCollector
    ) -> None:
        """Gauge recording should work."""
        metrics.record_gauge("skill.q_value", value=0.85)
        points = metrics.get_metric_points("skill.q_value")
        assert len(points) == 1
        assert points[0].value == 0.85

    def test_record_timing(
        self, metrics: MetricsCollector
    ) -> None:
        """Timing recording should work."""
        metrics.record_timing("skill.load.duration_ms", duration_ms=250.0)
        points = metrics.get_metric_points("skill.load.duration_ms")
        assert len(points) == 1

    def test_summarize(
        self, metrics: MetricsCollector
    ) -> None:
        """Summarize should compute correct statistics."""
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            metrics.record("latency", value=float(v))

        summary = metrics.summarize("latency")
        assert isinstance(summary, MetricSummary)
        assert summary.count == 10
        assert summary.sum == 550.0
        assert summary.min == 10.0
        assert summary.max == 100.0
        assert summary.mean == 55.0
        assert summary.p50 == pytest.approx(55.0, abs=5.0)
        assert summary.p95 == pytest.approx(95.5, abs=5.0)

    def test_summarize_empty(
        self, metrics: MetricsCollector
    ) -> None:
        """Summarizing a nonexistent metric should return zero stats."""
        summary = metrics.summarize("nonexistent")
        assert summary.count == 0

    def test_get_histogram(
        self, metrics: MetricsCollector
    ) -> None:
        """Histogram should bucket values correctly."""
        for v in [5, 15, 25, 50, 150, 500]:
            metrics.record("latency", value=float(v))

        histogram = metrics.get_histogram(
            "latency",
            buckets=[10.0, 50.0, 100.0, 500.0, float("inf")],
        )
        assert isinstance(histogram, dict)
        assert "<=10.0" in histogram
        assert histogram["<=10.0"] == 1  # 5ms

    def test_get_metric_names(
        self, metrics: MetricsCollector
    ) -> None:
        """Should list all unique metric names."""
        metrics.record("metric_a", value=1.0)
        metrics.record("metric_b", value=2.0)
        metrics.record("metric_a", value=3.0)

        names = metrics.get_metric_names()
        assert "metric_a" in names
        assert "metric_b" in names
        assert len(names) == 2

    def test_count(
        self, metrics: MetricsCollector
    ) -> None:
        """Count should return accurate totals."""
        assert metrics.count() == 0

        metrics.record("a", value=1.0)
        metrics.record("a", value=2.0)
        metrics.record("b", value=3.0)

        assert metrics.count() == 3
        assert metrics.count(name="a") == 2
        assert metrics.count(name="b") == 1

    def test_filter_by_labels(
        self, metrics: MetricsCollector
    ) -> None:
        """Filtering by labels should work."""
        metrics.record("latency", value=100.0, labels={"skill_id": "a"})
        metrics.record("latency", value=200.0, labels={"skill_id": "b"})

        points = metrics.get_metric_points(
            "latency", labels={"skill_id": "a"}
        )
        assert len(points) == 1
        assert points[0].value == 100.0


# =======================================================================
# TestStructuredLogger
# =======================================================================


class TestStructuredLogger:
    """Tests for the StructuredLogger module."""

    def test_log_and_retrieve(
        self, logger: StructuredLogger
    ) -> None:
        """Logging and retrieving should work."""
        entry = logger.info("Skill loaded", component="loader")
        assert isinstance(entry, LogEntry)
        assert entry.level == LogLevel.INFO
        assert entry.message == "Skill loaded"

        entries = logger.get_entries()
        assert len(entries) == 1

    def test_log_levels(
        self, logger: StructuredLogger
    ) -> None:
        """All log levels should be supported."""
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")

        assert logger.count() == 5
        assert logger.count(level=LogLevel.ERROR) == 1

    def test_log_with_metadata(
        self, logger: StructuredLogger
    ) -> None:
        """Logging with metadata should persist it."""
        entry = logger.info(
            "Skill executed",
            component="forge",
            skill_id="skill-1",
            metadata={"latency_ms": 150, "tokens": 500},
        )
        assert entry.metadata["latency_ms"] == 150

    def test_log_with_trace_correlation(
        self, logger: StructuredLogger
    ) -> None:
        """Logging with trace IDs should enable correlation."""
        entry = logger.log(
            level=LogLevel.INFO,
            message="Processing",
            component="engine",
            trace_id="trace-123",
            span_id="span-456",
        )

        entries = logger.get_entries(trace_id="trace-123")
        assert len(entries) == 1
        assert entries[0].span_id == "span-456"

    def test_get_entries_by_level(
        self, logger: StructuredLogger
    ) -> None:
        """Filtering by level should work."""
        logger.info("info")
        logger.error("error")
        logger.info("info2")

        errors = logger.get_entries(level=LogLevel.ERROR)
        assert len(errors) == 1
        assert errors[0].message == "error"

    def test_get_entries_by_component(
        self, logger: StructuredLogger
    ) -> None:
        """Filtering by component should work."""
        logger.info("msg1", component="loader")
        logger.info("msg2", component="tracker")

        loader_logs = logger.get_entries(component="loader")
        assert len(loader_logs) == 1

    def test_get_entries_by_skill(
        self, logger: StructuredLogger
    ) -> None:
        """Filtering by skill_id should work."""
        logger.info("msg1", skill_id="skill-a")
        logger.info("msg2", skill_id="skill-b")
        logger.info("msg3", skill_id="skill-a")

        a_logs = logger.get_entries(skill_id="skill-a")
        assert len(a_logs) == 2

    def test_search(
        self, logger: StructuredLogger
    ) -> None:
        """Search should find matching messages."""
        logger.info("Skill web-scraper loaded successfully")
        logger.info("Email sender failed with timeout")
        logger.info("Data parser started")

        results = logger.search("scraper")
        assert len(results) >= 1
        assert "scraper" in results[0].message.lower()

    def test_get_level_counts(
        self, logger: StructuredLogger
    ) -> None:
        """Level counts should be accurate."""
        logger.info("a")
        logger.info("b")
        logger.error("c")

        counts = logger.get_level_counts()
        assert counts["info"] == 2
        assert counts["error"] == 1

    def test_to_dict_and_json(
        self, logger: StructuredLogger
    ) -> None:
        """Serialization should work."""
        entry = logger.info("test msg", metadata={"key": "val"})
        d = entry.to_dict()
        assert d["message"] == "test msg"
        assert d["metadata"]["key"] == "val"

        j = entry.to_json()
        parsed = json.loads(j)
        assert parsed["message"] == "test msg"


# =======================================================================
# TestAsyncSupport
# =======================================================================


class TestAsyncSupport:
    """Tests for the async support modules."""

    @pytest.mark.asyncio
    async def test_async_registry_crud(
        self, db_dir: Path
    ) -> None:
        """Async registry CRUD operations should work."""
        async_reg = AsyncSkillRegistry(db_path=db_dir / "async_skills.db")

        # Register
        skill = await async_reg.register_skill(
            name="async-test",
            tier1_metadata="Async test skill",
            tags=["async"],
        )
        assert skill.name == "async-test"

        # Get
        fetched = await async_reg.get_skill(skill.id)
        assert fetched is not None
        assert fetched.name == "async-test"

        # Update
        updated = await async_reg.update_skill(
            skill.id, {"lifecycle": SkillLifecycle.ACTIVE}
        )
        assert updated is not None
        assert updated.lifecycle == SkillLifecycle.ACTIVE

        # List
        skills = await async_reg.list_skills()
        assert len(skills) == 1

        # Search
        results = await async_reg.search_skills("async")
        assert len(results) == 1

        await async_reg.close()

    @pytest.mark.asyncio
    async def test_async_tracker(
        self, db_dir: Path
    ) -> None:
        """Async tracker operations should work."""
        async_tracker = AsyncQValueTracker(db_path=db_dir / "async_tracker.db")

        # Record outcome
        outcome = Outcome(
            skill_id="skill-1",
            success=True,
            latency_ms=100.0,
            tokens_used=50,
        )
        await async_tracker.record_outcome(outcome)

        # Query
        q_value = await async_tracker.get_q_value("skill-1")
        assert isinstance(q_value, float)

        stats = await async_tracker.get_stats("skill-1")
        assert stats["total_outcomes"] == 1

        # TD update
        new_q = await async_tracker.td_lambda_update("skill-1", reward=1.0)
        assert new_q >= 0.5  # Should increase after success

        await async_tracker.close()

    @pytest.mark.asyncio
    async def test_async_loader(
        self, db_dir: Path
    ) -> None:
        """Async loader operations should work."""
        async_reg = AsyncSkillRegistry(db_path=db_dir / "async_loader_skills.db")
        async_tracker = AsyncQValueTracker(db_path=db_dir / "async_loader_tracker.db")

        # Register skills
        await async_reg.register_skill(
            name="web-scraper",
            tier1_metadata="Web scraping skill",
            tags=["web"],
            skill_id="skill-web",
        )
        await async_reg.update_skill("skill-web", {"lifecycle": SkillLifecycle.ACTIVE})

        # Load
        loader = AsyncProgressiveLoader(async_reg, async_tracker)
        skills = await loader.load_skill("web scraping")
        assert isinstance(skills, list)

        await async_reg.close()
        await async_tracker.close()


# =======================================================================
# TestVersioning
# =======================================================================


class TestSemanticVersion:
    """Tests for the SemanticVersion class."""

    def test_parse_basic(
        self,
    ) -> None:
        """Parsing a basic version should work."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.pre == ""
        assert v.build == ""

    def test_parse_with_prerelease(
        self,
    ) -> None:
        """Parsing with pre-release should work."""
        v = SemanticVersion.parse("1.0.0-alpha.1")
        assert v.pre == "alpha.1"

    def test_parse_with_build(
        self,
    ) -> None:
        """Parsing with build metadata should work."""
        v = SemanticVersion.parse("1.0.0+build.123")
        assert v.build == "build.123"

    def test_parse_full(
        self,
    ) -> None:
        """Parsing a full version should work."""
        v = SemanticVersion.parse("2.1.0-beta.2+sha.abc123")
        assert v.major == 2
        assert v.minor == 1
        assert v.patch == 0
        assert v.pre == "beta.2"
        assert v.build == "sha.abc123"

    def test_parse_invalid(
        self,
    ) -> None:
        """Parsing an invalid version should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            SemanticVersion.parse("not.a.version")

    def test_str(
        self,
    ) -> None:
        """String representation should be canonical."""
        v = SemanticVersion(1, 2, 3, "beta", "build")
        assert str(v) == "1.2.3-beta+build"

    def test_comparison(
        self,
    ) -> None:
        """Version comparison should follow semver rules."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        v3 = SemanticVersion(1, 1, 0)
        v4 = SemanticVersion(1, 0, 1)

        assert v1 < v2
        assert v1 < v3
        assert v1 < v4
        assert v2 > v3
        assert v3 > v4

    def test_equality(
        self,
    ) -> None:
        """Equal versions should be equal."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0)
        assert v1 == v2

    def test_bump_major(
        self,
    ) -> None:
        """Bumping major should reset minor and patch."""
        v = SemanticVersion(1, 5, 3)
        bumped = v.bump(ChangeType.MAJOR)
        assert bumped == SemanticVersion(2, 0, 0)

    def test_bump_minor(
        self,
    ) -> None:
        """Bumping minor should reset patch."""
        v = SemanticVersion(1, 5, 3)
        bumped = v.bump(ChangeType.MINOR)
        assert bumped == SemanticVersion(1, 6, 0)

    def test_bump_patch(
        self,
    ) -> None:
        """Bumping patch should increment only patch."""
        v = SemanticVersion(1, 5, 3)
        bumped = v.bump(ChangeType.PATCH)
        assert bumped == SemanticVersion(1, 5, 4)


class TestVersionManager:
    """Tests for the VersionManager module."""

    def test_record_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Recording a version should persist it."""
        record = version_mgr.record_version(
            skill_id="skill-1",
            version="1.0.0",
            change_type=ChangeType.MINOR,
            changelog="Initial release",
            snapshot={"tier1_metadata": "Test skill"},
        )
        assert isinstance(record, VersionRecord)
        assert record.version == "1.0.0"
        assert record.skill_id == "skill-1"

    def test_record_invalid_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Recording an invalid version should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            version_mgr.record_version(
                skill_id="skill-1",
                version="bad-version",
            )

    def test_bump_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Bumping should auto-increment from the latest version."""
        version_mgr.record_version(
            skill_id="skill-1", version="1.0.0",
            change_type=ChangeType.MINOR,
        )

        record = version_mgr.bump_version(
            skill_id="skill-1",
            change_type=ChangeType.MINOR,
            changelog="Added feature",
        )
        assert record.version == "1.1.0"

    def test_bump_version_from_scratch(
        self, version_mgr: VersionManager
    ) -> None:
        """Bumping from scratch should start at 0.1.0."""
        record = version_mgr.bump_version(
            skill_id="new-skill",
            change_type=ChangeType.MINOR,
        )
        assert record.version == "0.1.0"

    def test_get_latest_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Getting the latest version should return the most recent."""
        version_mgr.record_version(skill_id="s", version="1.0.0")
        version_mgr.record_version(skill_id="s", version="1.1.0")
        version_mgr.record_version(skill_id="s", version="1.2.0")

        latest = version_mgr.get_latest_version("s")
        assert latest is not None
        assert latest.version == "1.2.0"

    def test_get_latest_version_none(
        self, version_mgr: VersionManager
    ) -> None:
        """Getting latest when no versions exist should return None."""
        assert version_mgr.get_latest_version("nonexistent") is None

    def test_get_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Getting a specific version should work."""
        version_mgr.record_version(
            skill_id="s", version="1.0.0",
            snapshot={"data": "v1"},
        )
        version_mgr.record_version(
            skill_id="s", version="2.0.0",
            snapshot={"data": "v2"},
        )

        v1 = version_mgr.get_version("s", "1.0.0")
        assert v1 is not None
        assert v1.snapshot["data"] == "v1"

    def test_get_history(
        self, version_mgr: VersionManager
    ) -> None:
        """History should return all versions, most recent first."""
        version_mgr.record_version(skill_id="s", version="1.0.0")
        version_mgr.record_version(skill_id="s", version="1.1.0")
        version_mgr.record_version(skill_id="s", version="2.0.0")

        history = version_mgr.get_history("s")
        assert len(history) == 3
        assert history[0].version == "2.0.0"
        assert history[2].version == "1.0.0"

    def test_rollback(
        self, version_mgr: VersionManager
    ) -> None:
        """Rollback should create a new record with the old snapshot."""
        version_mgr.record_version(
            skill_id="s", version="1.0.0",
            snapshot={"instructions": "original"},
            changelog="Initial",
        )
        version_mgr.record_version(
            skill_id="s", version="2.0.0",
            snapshot={"instructions": "breaking change"},
            changelog="Breaking",
        )

        rolled_back = version_mgr.rollback("s", "1.0.0")
        assert rolled_back is not None
        assert rolled_back.snapshot["instructions"] == "original"
        assert "Rollback" in rolled_back.changelog

    def test_rollback_nonexistent(
        self, version_mgr: VersionManager
    ) -> None:
        """Rolling back to a nonexistent version should return None."""
        assert version_mgr.rollback("s", "99.99.99") is None

    def test_diff(
        self, version_mgr: VersionManager
    ) -> None:
        """Diff should show additions, removals, and modifications."""
        version_mgr.record_version(
            skill_id="s", version="1.0.0",
            snapshot={
                "name": "old-name",
                "removed_field": "gone",
                "unchanged": "same",
            },
        )
        version_mgr.record_version(
            skill_id="s", version="2.0.0",
            snapshot={
                "name": "new-name",
                "unchanged": "same",
                "new_field": "added",
            },
        )

        diff = version_mgr.diff("s", "1.0.0", "2.0.0")
        assert isinstance(diff, VersionDiff)
        assert diff.has_changes
        assert "new_field" in diff.added
        assert "removed_field" in diff.removed
        assert "name" in diff.modified
        assert diff.modified["name"]["from"] == "old-name"
        assert diff.modified["name"]["to"] == "new-name"

    def test_diff_no_changes(
        self, version_mgr: VersionManager
    ) -> None:
        """Diff with identical snapshots should have no changes."""
        snap = {"a": 1, "b": 2}
        version_mgr.record_version(skill_id="s", version="1.0.0", snapshot=snap)
        version_mgr.record_version(skill_id="s", version="1.0.1", snapshot=snap)

        diff = version_mgr.diff("s", "1.0.0", "1.0.1")
        assert not diff.has_changes
        assert "no changes" in diff.summary

    def test_diff_nonexistent_version(
        self, version_mgr: VersionManager
    ) -> None:
        """Diffing against a nonexistent version should raise."""
        version_mgr.record_version(skill_id="s", version="1.0.0")
        with pytest.raises(ValueError, match="not found"):
            version_mgr.diff("s", "1.0.0", "99.0.0")

    def test_diff_latest(
        self, version_mgr: VersionManager
    ) -> None:
        """diff_latest should compare latest against a specified version."""
        version_mgr.record_version(
            skill_id="s", version="1.0.0",
            snapshot={"x": 1},
        )
        version_mgr.record_version(
            skill_id="s", version="2.0.0",
            snapshot={"x": 2, "y": 3},
        )

        diff = version_mgr.diff_latest("s", "1.0.0")
        assert diff.from_version == "1.0.0"
        assert diff.to_version == "2.0.0"
        assert "y" in diff.added
        assert "x" in diff.modified

    def test_get_changelog(
        self, version_mgr: VersionManager
    ) -> None:
        """Changelog should return formatted version history."""
        version_mgr.record_version(
            skill_id="s", version="1.0.0",
            change_type=ChangeType.MINOR,
            changelog="Initial release",
            author="alice",
        )
        version_mgr.record_version(
            skill_id="s", version="1.1.0",
            change_type=ChangeType.MINOR,
            changelog="Added new feature",
            author="bob",
        )

        changelog = version_mgr.get_changelog("s")
        assert len(changelog) == 2
        assert changelog[0]["version"] == "1.1.0"
        assert changelog[0]["author"] == "bob"

    def test_count(
        self, version_mgr: VersionManager
    ) -> None:
        """Count should return accurate totals."""
        assert version_mgr.count() == 0

        version_mgr.record_version(skill_id="s1", version="1.0.0")
        version_mgr.record_version(skill_id="s1", version="1.1.0")
        version_mgr.record_version(skill_id="s2", version="1.0.0")

        assert version_mgr.count() == 3
        assert version_mgr.count(skill_id="s1") == 2
        assert version_mgr.count(skill_id="s2") == 1
