"""SkillForge MCP (Model Context Protocol) server.

Exposes SkillForge tools over stdin/stdout JSON-RPC so that any MCP-compatible
agent (e.g. Hermes Agent, Claude Desktop) can load skills, record outcomes,
run evolution, and inspect diagnostics — all through the standard MCP
tool-call interface.

Usage::

    from skillforge.mcp.server import SkillForgeMCPServer

    server = SkillForgeMCPServer("~/.skillforge/skillforge.db")
    server.run()

Or from the command line::

    python -m skillforge.mcp.server
"""
