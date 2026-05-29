#!/usr/bin/env python3
"""
Hermes ↔ SkillForge Integration Example

Demonstrates the full lifecycle of integrating Hermes Agent skills with
SkillForge: importing from Hermes, registering custom skills, tracking
outcomes, managing dependencies, and exporting back to Hermes format.

Usage:
    python -m examples.hermes_integration.example
    # or
    python examples/hermes_integration/example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path for direct execution
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.registry import SkillLifecycle, SkillRegistry
from skillforge.core.tracker import Outcome, QValueTracker
from skillforge.core.loader import ProgressiveLoader
from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter


def create_sample_hermes_skills(base_dir: Path) -> None:
    """Create sample SKILL.md files simulating a Hermes skills directory."""

    skills = {
        "email/himalaya": {
            "name": "himalaya",
            "description": "Himalaya CLI: IMAP/SMTP email from terminal.",
            "version": "1.1.0",
            "author": "community",
            "tags": ["Email", "IMAP", "SMTP", "CLI"],
            "body": (
                "# Himalaya Email CLI\n\n"
                "Himalaya is a CLI email client for managing emails "
                "from the terminal using IMAP and SMTP.\n"
            ),
        },
        "social-media/xurl": {
            "name": "xurl",
            "description": "X/Twitter via xurl CLI: post, search, DM, media.",
            "version": "1.1.1",
            "author": "xdevplatform",
            "tags": ["twitter", "x", "social-media", "xurl"],
            "body": (
                "# xurl — X (Twitter) API via the Official CLI\n\n"
                "xurl is the X developer platform's official CLI for the X API.\n"
            ),
        },
        "devops/kanban-worker": {
            "name": "kanban-worker",
            "description": "Kanban board automation for development workflows.",
            "version": "1.0.0",
            "author": "hermes-agent",
            "tags": ["kanban", "devops", "automation"],
            "body": (
                "# Kanban Worker\n\n"
                "Automates kanban board management for development teams.\n"
            ),
        },
    }

    for rel_path, info in skills.items():
        skill_dir = base_dir / rel_path
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        tags_str = ", ".join(info["tags"])
        skill_md.write_text(
            f"---\n"
            f"name: {info['name']}\n"
            f'description: "{info["description"]}"\n'
            f"version: {info['version']}\n"
            f"author: {info['author']}\n"
            f"metadata:\n"
            f"  hermes:\n"
            f"    tags: [{tags_str}]\n"
            f"---\n\n"
            f"{info['body']}",
            encoding="utf-8",
        )

    print(f"Created {len(skills)} sample Hermes skills in {base_dir}")


def main() -> None:
    """Run the full integration example."""

    with tempfile.TemporaryDirectory(prefix="skillforge_hermes_") as tmpdir:
        tmpdir = Path(tmpdir)

        # ────────────────────────────────────────────────────────
        # Step 1: Set up the environment
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 1: Setting up SkillForge + Hermes environment")
        print("=" * 60)

        hermes_dir = tmpdir / "hermes_skills"
        hermes_dir.mkdir()
        create_sample_hermes_skills(hermes_dir)

        db_path = tmpdir / "skills.db"
        tracker_path = tmpdir / "tracker.db"

        registry = SkillRegistry(db_path=db_path)
        tracker = QValueTracker(db_path=tracker_path)
        graph = SkillDependencyGraph()

        adapter = HermesSkillForgeAdapter(
            skillforge=registry,
            hermes_skills_dir=str(hermes_dir),
        )
        print(f"  Registry DB: {db_path}")
        print(f"  Tracker DB:  {tracker_path}")
        print(f"  Hermes dir:  {hermes_dir}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 2: Import Hermes skills into SkillForge
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 2: Importing Hermes skills into SkillForge")
        print("=" * 60)

        imported = adapter.import_hermes_skills()
        for skill in imported:
            print(f"  ✓ Imported: {skill.name} (v{skill.version}, tags={skill.tags})")

        print(f"\n  Total skills in registry: {len(registry.list_skills())}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 3: Register custom SkillForge skills
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 3: Registering custom SkillForge skills")
        print("=" * 60)

        http_client = registry.register_skill(
            name="http-client",
            tier1_metadata="Generic HTTP client for API calls",
            tier2_core="Use Python requests or httpx for API communication.",
            tags=["http", "api", "networking"],
        )
        registry.update_skill(http_client.id, {"lifecycle": SkillLifecycle.ACTIVE})
        print(f"  ✓ Registered: {http_client.name} ({http_client.id})")

        data_pipeline = registry.register_skill(
            name="data-pipeline",
            tier1_metadata="ETL data pipeline for processing data",
            tier2_core="Orchestrate data extraction, transformation, and loading.",
            tags=["etl", "data", "pipeline"],
        )
        registry.update_skill(data_pipeline.id, {"lifecycle": SkillLifecycle.ACTIVE})
        print(f"  ✓ Registered: {data_pipeline.name} ({data_pipeline.id})")
        print()

        # ────────────────────────────────────────────────────────
        # Step 4: Build dependency graph
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 4: Building skill dependency graph")
        print("=" * 60)

        # Add all skills to the graph
        for skill in registry.list_skills():
            graph.add_skill(skill.id)

        # Set up dependencies
        graph.add_dependency(http_client.id, "hermes-himalaya", weight=0.3)
        graph.add_dependency(data_pipeline.id, http_client.id, weight=0.7)
        graph.add_dependency("hermes-kanban-worker", data_pipeline.id, weight=0.5)

        for skill_id in graph.get_skills():
            deps = graph.get_dependencies(skill_id)
            if deps:
                for dep_id, weight in deps:
                    print(f"  {skill_id} → {dep_id} (weight={weight})")

        print(f"\n  Topological order: {graph.topological_sort()}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 5: Simulate usage and track outcomes
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 5: Simulating skill usage and tracking outcomes")
        print("=" * 60)

        # Simulate email skill: mostly successful
        for _ in range(15):
            tracker.record_outcome(
                Outcome(
                    skill_id="hermes-himalaya",
                    success=True,
                    latency_ms=120.0,
                    tokens_used=200,
                )
            )
        tracker.record_outcome(
            Outcome(
                skill_id="hermes-himalaya",
                success=False,
                latency_ms=500.0,
                tokens_used=400,
            )
        )

        # Simulate Twitter skill: mixed results
        for i in range(10):
            tracker.record_outcome(
                Outcome(
                    skill_id="hermes-xurl",
                    success=i % 3 != 0,
                    latency_ms=200.0 + (i * 10),
                    tokens_used=150,
                )
            )

        # Simulate HTTP client: very reliable
        for _ in range(20):
            tracker.record_outcome(
                Outcome(
                    skill_id=http_client.id,
                    success=True,
                    latency_ms=80.0,
                    tokens_used=100,
                )
            )

        # Apply TD(λ) updates
        for skill_id in ["hermes-himalaya", "hermes-xurl", http_client.id]:
            stats = tracker.get_stats(skill_id)
            reward = stats["success_rate"]
            new_q = tracker.td_lambda_update(skill_id, reward=reward)

            # Update registry with new Q-value
            registry.update_skill(
                skill_id,
                {
                    "q_value": new_q,
                    "success_rate": stats["success_rate"],
                    "usage_count": stats["total_outcomes"],
                },
            )

            print(
                f"  {skill_id}: "
                f"Q={new_q:.3f}, "
                f"success_rate={stats['success_rate']:.2f}, "
                f"avg_latency={stats['avg_latency_ms']:.0f}ms"
            )
        print()

        # ────────────────────────────────────────────────────────
        # Step 6: Q-value propagation through the graph
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 6: Propagating Q-value updates through dependencies")
        print("=" * 60)

        propagated = graph.propagate_q_update(
            http_client.id, delta=0.2, gamma=0.8
        )
        for affected_id, delta in propagated.items():
            print(f"  {affected_id}: ΔQ = {delta:+.4f}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 7: Progressive loading with routing
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 7: Progressive skill loading and routing")
        print("=" * 60)

        loader = ProgressiveLoader(registry, tracker)

        for strategy in ("q_value", "relevance", "success_rate"):
            results = loader.load_skill("email", routing=strategy, tier=2)
            names = [s.name for s in results]
            print(f"  Strategy '{strategy}': {names}")

        # Sticky skills
        sticky = loader.get_sticky_skills(limit=3)
        print(f"\n  Sticky skills: {[s.name for s in sticky]}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 8: Export back to Hermes format
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 8: Exporting SkillForge skills to Hermes format")
        print("=" * 60)

        for skill in [http_client, data_pipeline]:
            path = adapter.export_skill_to_hermes(skill.id, category="skillforge")
            if path:
                print(f"  ✓ Exported {skill.name} → {path}")

        # ────────────────────────────────────────────────────────
        # Step 9: Full bidirectional sync
        # ────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("Step 9: Running bidirectional sync")
        print("=" * 60)

        summary = adapter.sync()
        print(f"  Imported: {summary['imported']}")
        print(f"  Exported: {summary['exported']}")
        print(f"  Errors:   {len(summary['errors'])}")
        print()

        # ────────────────────────────────────────────────────────
        # Step 10: Final summary
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("Step 10: Final state summary")
        print("=" * 60)

        all_skills = registry.list_skills()
        print(f"\n  Total skills: {len(all_skills)}")
        print(f"  Graph nodes:  {len(graph)}")
        print(f"  Graph edges:  {sum(len(e) for e in graph._adjacency.values())}")
        print()

        print("  Skill details:")
        for skill in sorted(all_skills, key=lambda s: s.q_value, reverse=True):
            stats = tracker.get_stats(skill.id)
            print(
                f"    {skill.name:20s} | "
                f"Q={skill.q_value:.3f} | "
                f"success={stats['success_rate']:.2f} | "
                f"uses={stats['total_outcomes']:3d} | "
                f"lifecycle={skill.lifecycle.value}"
            )

        print()
        print("✓ Integration example complete!")

        # Cleanup
        registry.close()
        tracker.close()


if __name__ == "__main__":
    main()
