"""SkillForge command-line interface.

Provides a unified CLI built on ``argparse`` for managing skills, running
the API server, and inspecting system health — all using the Python standard
library only.

Usage::

    # Start the API server
    skillforge serve --host 0.0.0.0 --port 9000

    # List registered skills
    skillforge skills list

    # Get skill detail
    skillforge skills get <skill-id>

    # Search for skills
    skillforge skills search "greeting"

    # Trigger evolution
    skillforge evolve

    # Health check
    skillforge health

    # Dashboard summary
    skillforge dashboard

    # Import skills from Hermes
    skillforge import-hermes --path ~/.hermes
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from typing import Any

from .client import SkillForgeClient, SkillForgeClientError


# ---------------------------------------------------------------------------
# Table / formatting helpers (stdlib-only)
# ---------------------------------------------------------------------------


def _trunc(text: str, width: int) -> str:
    """Truncate *text* to *width* characters, appending ``…`` if needed."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print an ASCII table to stdout.

    Parameters
    ----------
    headers : list[str]
        Column headers.
    rows : list[list[str]]
        One inner list per row; lengths must match *headers*.
    """
    if not rows:
        print("  (no data)")
        return

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    # Header
    header_line = "  ".join(
        h.ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    separator = "  ".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)

    # Data rows
    for row in rows:
        line = "  ".join(
            (row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(headers))
        )
        print(line)


def _pretty_json(data: Any) -> None:
    """Pretty-print JSON data to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _format_q(q_value: float) -> str:
    """Format a Q-value as a percentage-like string."""
    return f"{q_value:.3f}"


def _format_rate(rate: float) -> str:
    """Format a success rate as a percentage."""
    return f"{rate * 100:.1f}%"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the SkillForge API server.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (host, port, db_path).

    Returns
    -------
    int
        Exit code (never returns in normal operation).
    """
    from skillforge.forge import SkillForge
    from skillforge.api.server import SkillForgeAPIServer

    db_path = args.db_path or "~/.skillforge/skillforge.db"
    forge = SkillForge(db_path=db_path)
    server = SkillForgeAPIServer(forge, host=args.host, port=args.port)

    def _shutdown(signum: int, frame: Any) -> None:
        print("\nShutting down...")
        server.stop()
        forge.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()
    print(f"SkillForge API listening on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    # Block main thread until interrupted
    try:
        while server.is_running():
            signal.pause()
    except AttributeError:
        # signal.pause() not available on Windows; fall back to sleep loop
        import time
        while server.is_running():
            time.sleep(1)
    return 0


def cmd_skills_list(args: argparse.Namespace) -> int:
    """List all skills.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        skills = client.list_skills()
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    if args.json:
        _pretty_json(skills)
        return 0

    headers = ["ID", "Name", "Lifecycle", "Q-Value", "Success%", "Usage", "Tags"]
    rows: list[list[str]] = []
    for s in skills:
        rows.append([
            _trunc(s.get("id", ""), 12),
            _trunc(s.get("name", ""), 25),
            s.get("lifecycle", "?"),
            _format_q(s.get("q_value", 0)),
            _format_rate(s.get("success_rate", 0)),
            str(s.get("usage_count", 0)),
            ", ".join(s.get("tags", [])),
        ])
    _print_table(headers, rows)
    print(f"\nTotal: {len(skills)} skill(s)")
    return 0


def cmd_skills_get(args: argparse.Namespace) -> int:
    """Get detail for a single skill.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (id).

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        skill = client.get_skill(args.id)
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    _pretty_json(skill)
    return 0


def cmd_skills_search(args: argparse.Namespace) -> int:
    """Search for skills by query.

    Uses the progressive loader endpoint with tier=1.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (query).

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        result = client.load_skill(query=args.query, tier=1)
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    skills = result.get("skills", [])

    if args.json:
        _pretty_json(skills)
        return 0

    if not skills:
        print("No skills found matching query.")
        return 0

    headers = ["ID", "Name", "Q-Value", "Success%", "Tags"]
    rows: list[list[str]] = []
    for s in skills:
        rows.append([
            _trunc(s.get("id", ""), 12),
            _trunc(s.get("name", ""), 30),
            _format_q(s.get("q_value", 0)),
            _format_rate(s.get("success_rate", 0)),
            ", ".join(s.get("tags", [])),
        ])
    _print_table(headers, rows)
    print(f"\nFound: {len(skills)} skill(s)")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """Trigger an evolution cycle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        result = client.run_evolution()
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    if args.json:
        _pretty_json(result)
        return 0

    report = result.get("report", result)
    print("=== Evolution Cycle Complete ===")
    print(f"  Skills evaluated: {report.get('total_skills_evaluated', 0)}")
    print(f"  Healthy:   {report.get('skills_healthy', 0)}")
    print(f"  Warning:   {report.get('skills_warning', 0)}")
    print(f"  Critical:  {report.get('skills_critical', 0)}")
    print(f"  Evolved:   {report.get('skills_evolved', 0)}")
    print(f"  Pruned:    {report.get('skills_pruned', 0)}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check API server health.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        result = client.get_health()
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    if args.json:
        _pretty_json(result)
        return 0

    print(f"Status:       {result.get('status', 'unknown')}")
    print(f"Server:       {result.get('server', '?')} v{result.get('version', '?')}")
    print(f"Total skills: {result.get('total_skills', 0)}")
    print(f"Timestamp:    {result.get('timestamp', '?')}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Show dashboard summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    client = SkillForgeClient(base_url=args.api_url)
    try:
        result = client.get_dashboard()
    except SkillForgeClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1

    if args.json:
        _pretty_json(result)
        return 0

    print("=== SkillForge Dashboard ===")
    print(f"  Total skills:   {result.get('total_skills', 0)}")
    print(f"  Healthy:        {result.get('healthy', 0)}")
    print(f"  Warning:        {result.get('warning', 0)}")
    print(f"  Critical:       {result.get('critical', 0)}")
    print(f"  Avg Q-value:    {result.get('average_q_value', 0):.4f}")
    print(f"  Total outcomes: {result.get('total_outcomes', 0)}")
    return 0


def cmd_import_hermes(args: argparse.Namespace) -> int:
    """Import skills from a Hermes Agent skills directory.

    Scans ``<path>/skills/`` for ``SKILL.md`` files and registers each as
    a SkillForge skill via the API.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (path).

    Returns
    -------
    int
        Exit code.
    """
    import os
    from pathlib import Path

    hermes_path = Path(args.path).expanduser().resolve()
    skills_dir = hermes_path / "skills"

    if not skills_dir.is_dir():
        print(f"Error: Skills directory not found: {skills_dir}", file=sys.stderr)
        return 1

    client = SkillForgeClient(base_url=args.api_url)
    imported = 0
    errors = 0

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue

        content = skill_md.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Extract name from first heading or directory name
        name = skill_dir.name
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                name = stripped[2:].strip()
                break

        # First paragraph as description, rest as instructions
        paragraphs = content.split("\n\n")
        description = paragraphs[0].strip() if paragraphs else name
        if description.startswith("# "):
            description = description[2:].strip()
        instructions = "\n\n".join(paragraphs[1:]).strip() if len(paragraphs) > 1 else ""

        try:
            client.create_skill(
                name=name,
                description=description[:200],  # Truncate for tier1
                instructions=instructions or content,
                tags=["hermes", skill_dir.name],
                skill_id=f"hermes:{skill_dir.name}",
            )
            imported += 1
            print(f"  ✓ Imported: {name}")
        except SkillForgeClientError as exc:
            if "already exists" in exc.detail.lower():
                print(f"  · Skipped (exists): {name}")
            else:
                print(f"  ✗ Failed: {name} — {exc.detail}", file=sys.stderr)
                errors += 1

    print(f"\nImported: {imported} skill(s), Errors: {errors}")
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all sub-commands.

    Returns
    -------
    argparse.ArgumentParser
        The fully-configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="skillforge",
        description="SkillForge — adaptive skill management for AI agents",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8742",
        help="SkillForge API base URL (default: http://127.0.0.1:8742)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON instead of formatted tables",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8742, help="Port (default: 8742)")
    p_serve.add_argument("--db-path", default=None, help="SQLite database path")
    p_serve.set_defaults(func=cmd_serve)

    # --- skills ---
    p_skills = subparsers.add_parser("skills", help="Manage skills")
    skills_sub = p_skills.add_subparsers(dest="skills_cmd", help="Skill commands")

    # skills list
    p_list = skills_sub.add_parser("list", help="List all skills")
    p_list.set_defaults(func=cmd_skills_list)

    # skills get
    p_get = skills_sub.add_parser("get", help="Get skill detail")
    p_get.add_argument("id", help="Skill ID")
    p_get.set_defaults(func=cmd_skills_get)

    # skills search
    p_search = skills_sub.add_parser("search", help="Search for skills")
    p_search.add_argument("query", help="Search query")
    p_search.set_defaults(func=cmd_skills_search)

    # --- evolve ---
    p_evolve = subparsers.add_parser("evolve", help="Trigger evolution cycle")
    p_evolve.set_defaults(func=cmd_evolve)

    # --- health ---
    p_health = subparsers.add_parser("health", help="Check API health")
    p_health.set_defaults(func=cmd_health)

    # --- dashboard ---
    p_dash = subparsers.add_parser("dashboard", help="Dashboard summary")
    p_dash.set_defaults(func=cmd_dashboard)

    # --- import-hermes ---
    p_import = subparsers.add_parser("import-hermes", help="Import Hermes skills")
    p_import.add_argument(
        "--path",
        default="~/.hermes",
        help="Hermes Agent home directory (default: ~/.hermes)",
    )
    p_import.set_defaults(func=cmd_import_hermes)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Handle nested sub-commands (e.g. "skills list")
    if args.command == "skills" and not getattr(args, "skills_cmd", None):
        parser.parse_args(["skills", "--help"])
        return 0

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
