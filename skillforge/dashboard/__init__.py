"""SkillForge Web Dashboard.

A self-contained web dashboard for monitoring skill health, Q-values,
success rates, and evolution status.  Built on ``http.server`` with no
external dependencies.

Usage::

    from skillforge.dashboard.app import DashboardServer

    dashboard = DashboardServer("~/.skillforge/skillforge.db")
    dashboard.start()
    # Open http://127.0.0.1:8743 in your browser
    dashboard.stop()
"""
