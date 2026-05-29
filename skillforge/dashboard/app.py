"""SkillForge web dashboard built on ``http.server``.

Provides a single-page web application for monitoring skill health, Q-values,
success rates, and evolution status.  Everything is served from a single
Python file — no external HTML/CSS/JS files needed.

Usage::

    from skillforge.dashboard.app import DashboardServer

    server = DashboardServer("~/.skillforge/skillforge.db")
    server.start()          # starts in a background thread
    # … browse to http://127.0.0.1:8743
    server.stop()

Or from the command line::

    python -m skillforge.dashboard.app
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from skillforge.forge import SkillForge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded HTML dashboard (single-page application)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML: str = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SkillForge Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#242837;--border:#2e3348;
  --text:#e4e6f0;--text2:#8b8fa8;--accent:#6c7bff;--accent2:#4c5bdb;
  --green:#34d399;--yellow:#fbbf24;--red:#f87171;--blue:#60a5fa;
  --radius:10px;--shadow:0 2px 12px rgba(0,0,0,.35);
}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}
a{color:var(--accent);text-decoration:none}

/* Layout */
.app{max-width:1400px;margin:0 auto;padding:24px 20px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;flex-wrap:wrap;gap:12px}
header h1{font-size:1.6rem;font-weight:700;letter-spacing:-.02em}
header h1 span{color:var(--accent)}
.status-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;
  border-radius:20px;font-size:.8rem;font-weight:600;background:var(--surface);border:1px solid var(--border)}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;box-shadow:var(--shadow);transition:transform .15s}
.card:hover{transform:translateY(-2px)}
.card-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:6px}
.card-value{font-size:2rem;font-weight:700}
.card-sub{font-size:.8rem;color:var(--text2);margin-top:4px}
.card-value.green{color:var(--green)}.card-value.yellow{color:var(--yellow)}
.card-value.red{color:var(--red)}.card-value.blue{color:var(--blue)}

/* Table */
.table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  overflow:hidden;box-shadow:var(--shadow)}
.table-header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;
  border-bottom:1px solid var(--border)}
.table-header h2{font-size:1.05rem;font-weight:600}
.table-header .actions{display:flex;gap:8px}
.btn{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface2);
  color:var(--text);font-size:.8rem;cursor:pointer;transition:all .15s}
.btn:hover{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.active{background:var(--accent);border-color:var(--accent);color:#fff}
table{width:100%;border-collapse:collapse}
thead{background:var(--surface2)}
th{padding:10px 16px;text-align:left;font-size:.75rem;text-transform:uppercase;
  letter-spacing:.06em;color:var(--text2);font-weight:600;border-bottom:1px solid var(--border)}
td{padding:12px 16px;border-bottom:1px solid var(--border);font-size:.88rem}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(108,123,255,.04)}

/* Badges */
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.72rem;font-weight:600}
.badge-healthy{background:rgba(52,211,153,.15);color:var(--green)}
.badge-warning{background:rgba(251,191,36,.15);color:var(--yellow)}
.badge-critical{background:rgba(248,113,113,.15);color:var(--red)}
.badge-dead{background:rgba(139,143,168,.15);color:var(--text2)}

/* Progress bar */
.progress{width:100%;height:6px;background:var(--surface2);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;border-radius:3px;transition:width .4s}

/* Responsive */
@media(max-width:768px){
  .cards{grid-template-columns:repeat(2,1fr)}
  th,td{padding:8px 10px;font-size:.8rem}
  .app{padding:16px 12px}
}
@media(max-width:480px){
  .cards{grid-template-columns:1fr}
}

/* Loading */
.loading{text-align:center;padding:40px;color:var(--text2)}
.spinner{display:inline-block;width:24px;height:24px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Footer */
footer{text-align:center;padding:20px;color:var(--text2);font-size:.75rem;margin-top:28px}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>⚡ <span>SkillForge</span> Dashboard</h1>
    <div class="status-badge">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Connecting…</span>
    </div>
  </header>

  <div class="cards" id="cards">
    <div class="loading"><div class="spinner"></div></div>
  </div>

  <div class="table-wrap">
    <div class="table-header">
      <h2>Skills</h2>
      <div class="actions">
        <button class="btn" id="btnRefresh" onclick="refresh()">↻ Refresh</button>
      </div>
    </div>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Q-Value</th>
          <th>Success</th>
          <th>Usage</th>
          <th>Health</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="skillsBody">
        <tr><td colspan="6" class="loading"><div class="spinner"></div></td></tr>
      </tbody>
    </table>
    </div>
  </div>

  <footer>SkillForge v0.4.0 — Auto-refresh every 10 seconds</footer>
</div>

<script>
const API = '';
const REFRESH_MS = 10000;
let refreshTimer = null;

async function fetchJSON(url) {
  const res = await fetch(API + url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function healthClass(q) {
  if (q >= 0.7) return 'healthy';
  if (q >= 0.4) return 'warning';
  if (q >= 0.2) return 'critical';
  return 'dead';
}

function healthLabel(q) {
  const cls = healthClass(q);
  const labels = {healthy:'Healthy',warning:'Warning',critical:'Critical',dead:'Dead'};
  return `<span class="badge badge-${cls}">${labels[cls]}</span>`;
}

function progressBar(value, max) {
  const pct = Math.min(100, (value / max) * 100);
  const color = value >= 0.7 ? 'var(--green)' : value >= 0.4 ? 'var(--yellow)' : 'var(--red)';
  return `<div class="progress"><div class="progress-fill" style="width:${pct}%;background:${color}"></div></div>`;
}

function renderCards(health, skills) {
  const total = health.total_skills || skills.length || 0;
  const avgQ = health.avg_health || 0;
  const status = health.by_status || {};
  const healthy = status.healthy || 0;
  const warning = status.warning || 0;
  const critical = (status.critical || 0) + (status.dead || 0);

  document.getElementById('cards').innerHTML = `
    <div class="card">
      <div class="card-label">Total Skills</div>
      <div class="card-value blue">${total}</div>
      <div class="card-sub">Active in registry</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Health</div>
      <div class="card-value ${healthClass(avgQ)}">${avgQ.toFixed(3)}</div>
      <div class="card-sub">${progressBar(avgQ, 1)}</div>
    </div>
    <div class="card">
      <div class="card-label">Healthy</div>
      <div class="card-value green">${healthy}</div>
      <div class="card-sub">${total?((healthy/total*100).toFixed(0)+'%'):'0%'}</div>
    </div>
    <div class="card">
      <div class="card-label">Warning</div>
      <div class="card-value yellow">${warning}</div>
      <div class="card-sub">${total?((warning/total*100).toFixed(0)+'%'):'0%'}</div>
    </div>
    <div class="card">
      <div class="card-label">Critical</div>
      <div class="card-value red">${critical}</div>
      <div class="card-sub">${total?((critical/total*100).toFixed(0)+'%'):'0%'}</div>
    </div>
  `;
}

function renderSkills(skills) {
  const tbody = document.getElementById('skillsBody');
  if (!skills.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text2);padding:30px">No skills registered yet</td></tr>';
    return;
  }
  tbody.innerHTML = skills.map(s => {
    const q = s.q_value != null ? s.q_value : 0.5;
    const sr = s.success_rate != null ? s.success_rate : 0.5;
    const lc = s.lifecycle || 'active';
    return `<tr>
      <td><strong>${esc(s.name || s.id)}</strong></td>
      <td>${q.toFixed(3)} ${progressBar(q, 1)}</td>
      <td>${(sr * 100).toFixed(1)}%</td>
      <td>${s.usage_count || 0}</td>
      <td>${healthLabel(q)}</td>
      <td><span class="badge badge-${lc==='active'?'healthy':lc==='deprecated'?'warning':'dead'}">${lc}</span></td>
    </tr>`;
  }).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

async function refresh() {
  try {
    const [skills, health] = await Promise.all([
      fetchJSON('/api/skills'),
      fetchJSON('/api/health'),
    ]);
    renderCards(health, skills);
    renderSkills(skills);
    document.getElementById('statusDot').style.background = 'var(--green)';
    document.getElementById('statusText').textContent = 'Live';
  } catch (e) {
    document.getElementById('statusDot').style.background = 'var(--red)';
    document.getElementById('statusText').textContent = 'Error';
    console.error('Refresh failed:', e);
  }
}

refresh();
refreshTimer = setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helper: serialise Skill objects for JSON
# ---------------------------------------------------------------------------

def _skill_to_dict(skill: Any) -> dict[str, Any]:
    """Convert a Skill dataclass to a JSON-safe dict."""
    d: dict[str, Any] = {}
    for key in ("id", "name", "version", "tier1_metadata", "q_value",
                "success_rate", "usage_count", "tags"):
        d[key] = getattr(skill, key, None)
    lifecycle = getattr(skill, "lifecycle", None)
    d["lifecycle"] = lifecycle.value if lifecycle is not None else "active"
    created = getattr(skill, "created_at", None)
    d["created_at"] = created.isoformat() if created is not None else None
    updated = getattr(skill, "updated_at", None)
    d["updated_at"] = updated.isoformat() if updated is not None else None
    return d


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard API and HTML page."""

    # Shared references set by DashboardServer.
    _db_path: str = ""
    _forge_lock: threading.Lock = threading.Lock()
    _thread_forge: Any = None

    @property
    def forge(self) -> Any:
        """Return (or lazily create) a thread-local SkillForge instance.

        SQLite connections cannot cross threads, so each HTTP handler thread
        gets its own forge backed by the same database file (WAL mode allows
        concurrent readers).
        """
        if _DashboardHandler._thread_forge is None:
            with _DashboardHandler._forge_lock:
                if _DashboardHandler._thread_forge is None:
                    from skillforge.forge import SkillForge
                    _DashboardHandler._thread_forge = SkillForge(
                        db_path=_DashboardHandler._db_path
                    )
        return _DashboardHandler._thread_forge

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/":
            self._serve_html()
        elif path == "/api/skills":
            self._serve_skills()
        elif path == "/api/stats":
            self._serve_stats()
        elif path == "/api/health":
            self._serve_health()
        elif path == "/api/evolution":
            self._serve_evolution()
        else:
            self._send_json({"error": "Not found"}, status=404)

    # -- route handlers -----------------------------------------------------

    def _serve_html(self) -> None:
        body = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_skills(self) -> None:
        try:
            skills = self.forge.list_skills()
            self._send_json([_skill_to_dict(s) for s in skills])
        except Exception as exc:
            logger.exception("Error serving /api/skills")
            self._send_json({"error": str(exc)}, status=500)

    def _serve_stats(self) -> None:
        try:
            stats = self.forge.get_skill_stats()
            self._send_json(stats)
        except Exception as exc:
            logger.exception("Error serving /api/stats")
            self._send_json({"error": str(exc)}, status=500)

    def _serve_health(self) -> None:
        try:
            from skillforge.intelligence.health_monitor import HealthMonitor

            monitor = HealthMonitor(
                registry=self.forge._registry,
                tracker=self.forge._tracker,
                graph=self.forge._graph,
            )
            summary = monitor.get_dashboard_summary()
            self._send_json(summary)
        except Exception:
            # Fallback: basic summary from skill stats.
            try:
                all_stats = self.forge.get_skill_stats()
                if not isinstance(all_stats, list):
                    all_stats = []
                total = len(all_stats)
                if total == 0:
                    self._send_json({
                        "total_skills": 0,
                        "avg_health": 0.0,
                        "by_status": {"healthy": 0, "warning": 0, "critical": 0},
                    })
                    return
                avg_q = sum(s.get("q_value", 0.5) for s in all_stats) / total
                healthy = sum(1 for s in all_stats if s.get("q_value", 0.5) >= 0.7)
                warning = sum(1 for s in all_stats if 0.4 <= s.get("q_value", 0.5) < 0.7)
                critical = sum(1 for s in all_stats if s.get("q_value", 0.5) < 0.4)
                self._send_json({
                    "total_skills": total,
                    "avg_health": round(avg_q, 4),
                    "by_status": {
                        "healthy": healthy,
                        "warning": warning,
                        "critical": critical,
                    },
                })
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

    def _serve_evolution(self) -> None:
        try:
            report = self.forge.run_evolution_loop()
            self._send_json(report.to_dict())
        except Exception as exc:
            logger.exception("Error serving /api/evolution")
            self._send_json({"error": str(exc)}, status=500)

    # -- helpers ------------------------------------------------------------

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging; use Python logger instead."""
        logger.debug(format, *args)


# ---------------------------------------------------------------------------
# DashboardServer
# ---------------------------------------------------------------------------

class DashboardServer:
    """Background web dashboard for SkillForge.

    Parameters
    ----------
    skillforge_or_db_path : SkillForge | str | Path
        Either an existing :class:`~skillforge.forge.SkillForge` instance or a
        path string to the SQLite database.
    host : str
        Bind address.  Default ``127.0.0.1``.
    port : int
        Listen port.  Default ``8743``.
    """

    def __init__(
        self,
        skillforge_or_db_path: Any = "~/.skillforge/skillforge.db",
        host: str = "127.0.0.1",
        port: int = 8743,
    ) -> None:
        self._forge_arg = skillforge_or_db_path
        self._host = host
        self._port = port
        self._forge: Any = None
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def forge(self) -> Any:
        """Return (and lazily create) the SkillForge instance."""
        if self._forge is None:
            from skillforge.forge import SkillForge

            if isinstance(self._forge_arg, (str, Path)):
                self._forge = SkillForge(db_path=str(self._forge_arg))
            else:
                self._forge = self._forge_arg
        return self._forge

    @property
    def url(self) -> str:
        """Return the dashboard URL."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the dashboard server in a background daemon thread."""
        if self._server is not None:
            logger.warning("Dashboard server already running at %s", self.url)
            return

        # Inject the db_path into the handler class so it can lazily create
        # a thread-local SkillForge instance (SQLite is thread-bound).
        if isinstance(self._forge_arg, (str, Path)):
            _DashboardHandler._db_path = str(Path(self._forge_arg).expanduser())
        else:
            # Assume it's a SkillForge instance — extract its db path.
            _DashboardHandler._db_path = str(self._forge_arg._db_path)

        # Reset any stale thread-local forge from a previous server run.
        _DashboardHandler._thread_forge = None

        self._server = HTTPServer(
            (self._host, self._port), _DashboardHandler
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="skillforge-dashboard",
        )
        self._thread.start()
        logger.info("Dashboard running at %s", self.url)

    def stop(self) -> None:
        """Shut down the dashboard server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
            logger.info("Dashboard stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the dashboard server in the foreground (Ctrl+C to stop)."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8743

    server = DashboardServer(host=host, port=port)
    server.start()
    print(f"SkillForge Dashboard running at {server.url}")
    print("Press Ctrl+C to stop.")
    try:
        server._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        print("\nShutting down…")
        server.stop()


if __name__ == "__main__":
    main()
