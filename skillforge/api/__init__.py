"""SkillForge REST API — stdlib-only HTTP server, client, and CLI.

This package provides a lightweight REST API for SkillForge built entirely
on the Python standard library (``http.server``, ``urllib``, ``argparse``).

Modules
-------
server
    HTTP API server using ``http.server`` with a threaded background mode.
client
    Python client that talks to the API over HTTP (``urllib.request``).
cli
    Command-line interface powered by ``argparse``.
"""

from .server import SkillForgeAPIServer
from .client import SkillForgeClient

__all__ = ["SkillForgeAPIServer", "SkillForgeClient"]
