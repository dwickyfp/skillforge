# SkillForge Integration Guide

A comprehensive guide for integrating SkillForge with AI agents, frameworks, and toolchains.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hermes Agent Integration](#hermes-agent-integration)
3. [OpenClaw Integration](#openclaw-integration)
4. [Custom Agent Integration](#custom-agent-integration)
5. [MCP Server Integration](#mcp-server-integration)
6. [Advanced Patterns](#advanced-patterns)

---

## Quick Start

Install SkillForge and set up your first skill in under 2 minutes:

```python
from skillforge import SkillForge

# Initialize with default settings (database at ~/.skillforge/skillforge.db)
forge = SkillForge()

# Register a skill
skill = forge.register_skill(
    name="code-reviewer",
    tier1_metadata="Review code for bugs, style issues, and improvements",
    tier2_core="You are a code reviewer. Analyze the provided code for:\n"
               "1. Bug detection\n2. Style compliance\n3. Performance issues",
    tags=["code", "review", "quality"],
)

# Load skills matching a query
skills = forge.load_skill("review my code", tier=2)
print(f"Found {len(skills)} matching skills")

# Record an outcome
forge.record_outcome(skill.id, success=True, latency_ms=250, tokens_used=500)

# Clean up
forge.close()
```

Or use the context manager:

```python
with SkillForge() as forge:
    forge.register_skill("greeting", "Respond to greetings")
    skills = forge.load_skill("hello world")
    forge.record_outcome(skills[0].id, success=True)
```

---

## Hermes Agent Integration

SkillForge provides a **bidirectional adapter** for Hermes Agent that reads and writes `SKILL.md` files (the [agentskills.io](https://agentskills.io) standard).

### Prerequisites

- Hermes Agent installed with skills at `~/.hermes/skills/`
- SkillForge installed (`pip install -e .` from the project root)

### Step 1: Basic Setup

```python
from skillforge import SkillForge
from skillforge.integrations.hermes.adapter import HermesSkillForgeAdapter

forge = SkillForge(db_path="~/.skillforge/hermes.db")

adapter = HermesSkillForgeAdapter(
    skillforge=forge._registry,         # The internal SkillRegistry
    hermes_skills_dir="~/.hermes/skills"
)
```

### Step 2: Import All Hermes Skills

```python
imported = adapter.import_hermes_skills()
for skill in imported:
    print(f"Imported: {skill.name} (v{skill.version})")
    print(f"  Tags: {skill.tags}")
    print(f"  ID: {skill.id}")
```

Each imported skill gets a deterministic ID of the form `hermes-<name>` to enable idempotent re-imports.

### Step 3: Bidirectional Sync

```python
summary = adapter.sync()
print(f"Imported: {summary['imported']}")
print(f"Exported: {summary['exported']}")
print(f"Errors:   {len(summary['errors'])}")
```

The `sync()` method:
1. **Imports** all `SKILL.md` files from the Hermes directory into SkillForge (upsert)
2. **Exports** any SkillForge skills tagged with `hermes_source` that are missing from disk

### Step 4: Export a SkillForge Skill to Hermes

```python
# Register a new skill in SkillForge
skill = forge.register_skill(
    name="api-tester",
    tier1_metadata="Automated API testing with assertions",
    tier2_core="Generate and run API tests using pytest and httpx...",
    tags=["api", "testing", "automation"],
)

# Export it as a Hermes SKILL.md
path = adapter.export_skill_to_hermes(skill.id, category="devtools")
print(f"Exported to: {path}")
# → ~/.hermes/skills/devtools/api-tester/SKILL.md
```

### Step 5: Track Outcomes for Evolved Skills

```python
# After using a Hermes skill, record the outcome
forge.record_outcome(
    skill_id="hermes-himalaya",
    success=True,
    latency_ms=180,
    tokens_used=320,
    user_feedback=4.5
)

# Run evolution to improve underperforming skills
report = forge.run_evolution_loop()
print(report.summary())
```

### Full Example

See `examples/hermes_integration/example.py` for a complete working demo that covers importing, custom skill registration, dependency graph building, outcome tracking, Q-value propagation, progressive loading, and export.

---

## OpenClaw Integration

Integrate SkillForge with OpenClaw agents via the skill dictionary interface.

### Step 1: Import OpenClaw Skill Definitions

OpenClaw uses a JSON-based skill format. Use `forge.import_skill()` to bring them in:

```python
from skillforge import SkillForge

forge = SkillForge()

# OpenClaw skill format
openclaw_skill = {
    "name": "web-search",
    "description": "Search the web and summarize results",
    "instructions": "Use the search tool to find relevant results, "
                    "then synthesize a concise summary with citations.",
    "resources": ["https://search.example.com/docs"],
    "tags": ["search", "web", "research"],
}

skill = forge.import_skill(openclaw_skill)
print(f"Imported: {skill.name} ({skill.id})")
```

### Step 2: Load Skills at Runtime

```python
# When the OpenClaw agent needs skills for a task:
skills = forge.load_skill(
    query="search the web for recent news",
    tier=2,           # Include core instructions
    routing="q_value", # Rank by quality
    limit=3,
)

for skill in skills:
    print(f"Skill: {skill.name}")
    print(f"  Instructions: {skill.tier2_core[:80]}...")
    print(f"  Q-value: {skill.q_value:.3f}")
```

### Step 3: Post-Execution Feedback Loop

```python
# After the agent executes using a skill:
forge.record_outcome(
    skill_id=skill.id,
    success=True,          # Did the task succeed?
    latency_ms=350,        # Wall-clock time
    tokens_used=800,       # LLM tokens consumed
    user_feedback=4.0,     # Optional user rating (0-5)
)
```

### Step 4: Bulk Import from OpenClaw Manifest

```python
import json

with open("openclaw_skills.json") as f:
    manifest = json.load(f)

for skill_def in manifest["skills"]:
    try:
        forge.import_skill(skill_def)
    except ValueError as e:
        print(f"Skipped: {e}")

print(f"Total skills: {len(forge.list_skills())}")
```

---

## Custom Agent Integration

Integrate SkillForge with any custom agent framework.

### Architecture Overview

```
┌─────────────────┐     query      ┌──────────────┐
│  Your Agent      │ ──────────────▶│  SkillForge  │
│                  │◀──────────────│              │
│  Execute task    │  ranked skills │  Registry    │
│  using skills    │                │  Tracker     │
│                  │─── outcome ───▶│  Evolution   │
│  Record results  │                │  Loader      │
└─────────────────┘                └──────────────┘
```

### Minimal Agent Integration

```python
from skillforge import SkillForge

class MyAgent:
    def __init__(self):
        self.forge = SkillForge(db_path="~/.skillforge/myagent.db")
    
    def handle_task(self, task_description: str) -> str:
        # 1. Load relevant skills
        skills = self.forge.load_skill(
            query=task_description,
            tier=2,
            routing="q_value",
            limit=3,
        )
        
        # 2. Build system prompt with skill instructions
        skill_context = "\n\n".join(
            f"## Skill: {s.name}\n{s.tier2_core}"
            for s in skills
        )
        
        # 3. Execute with your LLM
        result = self._call_llm(task_description, skill_context)
        
        # 4. Record the outcome
        for skill in skills:
            self.forge.record_outcome(
                skill_id=skill.id,
                success=self._evaluate_result(result),
                latency_ms=result.latency_ms,
                tokens_used=result.tokens,
            )
        
        return result.text
    
    def shutdown(self):
        self.forge.close()
```

### With LLM-Powered Diagnosis

```python
def my_llm_fn(prompt: str) -> str:
    """Your LLM wrapper function."""
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Pass the LLM function for advanced failure analysis
forge = SkillForge(
    db_path="~/.skillforge/advanced.db",
    llm_fn=my_llm_fn,
)
```

When `llm_fn` is provided, the `SelfDiagnosisEngine` uses it for sophisticated failure analysis instead of falling back to rule-based heuristics.

### Periodic Evolution

```python
import schedule
import time

def evolve_skills():
    report = forge.run_evolution_loop()
    print(f"Evolved: {report.skills_evolved}, Pruned: {report.skills_pruned}")

# Run evolution every 6 hours
schedule.every(6).hours.do(evolve_skills)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Using the Dependency Graph

```python
from skillforge.core.graph import SkillDependencyGraph

graph = SkillDependencyGraph()

# Register skills in the graph
for skill in forge.list_skills():
    graph.add_skill(skill.id, metadata={"name": skill.name})

# Declare dependencies
graph.add_dependency("api-client", "http-parser", weight=0.8)
graph.add_dependency("test-runner", "api-client", weight=0.5)

# Impact analysis: what breaks if http-parser changes?
impacted = graph.downstream_impact("http-parser")
print(f"Skills affected: {impacted}")

# Propagate Q-value improvements
propagated = graph.propagate_q_update("http-parser", delta=0.15, gamma=0.9)
for skill_id, delta in propagated.items():
    print(f"  {skill_id}: ΔQ = {delta:+.4f}")
```

---

## MCP Server Integration

Expose SkillForge as an MCP (Model Context Protocol) server for use with any MCP-compatible agent.

### MCP Server Implementation

```python
#!/usr/bin/env python3
"""SkillForge MCP Server - exposes skills via MCP protocol."""

from skillforge import SkillForge
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

forge = SkillForge()
server = Server("skillforge-mcp")

@server.tool()
def load_skills(query: str, tier: int = 2, limit: int = 5) -> str:
    """Load skills matching the given query.
    
    Args:
        query: Natural language description of needed skills
        tier: Detail level (1=metadata, 2=instructions, 3=full)
        limit: Maximum number of skills to return
    """
    skills = forge.load_skill(query=query, tier=tier, limit=limit)
    return json.dumps([
        {
            "id": s.id,
            "name": s.name,
            "q_value": round(s.q_value, 3),
            "instructions": s.tier2_core,
            "tags": s.tags,
        }
        for s in skills
    ], indent=2)

@server.tool()
def record_outcome(
    skill_id: str,
    success: bool,
    latency_ms: float = 0,
    tokens_used: int = 0,
) -> str:
    """Record the outcome of using a skill.
    
    Args:
        skill_id: The skill that was used
        success: Whether the task succeeded
        latency_ms: Execution time in milliseconds
        tokens_used: Number of LLM tokens consumed
    """
    forge.record_outcome(skill_id, success, latency_ms, tokens_used)
    return json.dumps({"status": "recorded", "skill_id": skill_id})

@server.tool()
def list_skills(lifecycle: str = "active") -> str:
    """List all registered skills.
    
    Args:
        lifecycle: Filter by lifecycle stage (draft, active, deprecated, archived)
    """
    skills = forge.list_skills(filters={"lifecycle": lifecycle})
    return json.dumps([
        {
            "id": s.id,
            "name": s.name,
            "q_value": round(s.q_value, 3),
            "usage_count": s.usage_count,
            "tags": s.tags,
        }
        for s in skills
    ], indent=2)

@server.tool()
def register_skill(
    name: str,
    description: str,
    instructions: str = "",
    tags: str = "",
) -> str:
    """Register a new skill.
    
    Args:
        name: Skill name
        description: Short description for routing
        instructions: Full instructions/prompt
        tags: Comma-separated tags
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    skill = forge.register_skill(
        name=name,
        tier1_metadata=description,
        tier2_core=instructions,
        tags=tag_list,
    )
    return json.dumps({"status": "registered", "id": skill.id, "name": skill.name})

@server.tool()
def get_skill_stats(skill_id: str = "") -> str:
    """Get performance statistics for a skill or all skills.
    
    Args:
        skill_id: Specific skill ID, or empty for all skills
    """
    stats = forge.get_skill_stats(skill_id if skill_id else None)
    return json.dumps(stats, indent=2, default=str)

@server.tool()
def run_evolution() -> str:
    """Run an evolution cycle to improve underperforming skills."""
    report = forge.run_evolution_loop()
    return json.dumps({
        "evaluated": report.total_skills_evaluated,
        "evolved": report.skills_evolved,
        "pruned": report.skills_pruned,
        "healthy": report.skills_healthy,
        "warning": report.skills_warning,
        "critical": report.skills_critical,
    })

if __name__ == "__main__":
    server.run()
```

### MCP Client Usage

```python
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client

async def use_skillforge_mcp():
    async with stdio_client("python", "skillforge_mcp_server.py") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Load skills
            result = await session.call_tool(
                "load_skills",
                arguments={"query": "code review", "tier": 2}
            )
            print(result)
            
            # Record outcome
            await session.call_tool(
                "record_outcome",
                arguments={
                    "skill_id": "abc-123",
                    "success": True,
                    "latency_ms": 200,
                    "tokens_used": 500,
                }
            )
```

### MCP Configuration for Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "skillforge": {
      "command": "python",
      "args": ["-m", "skillforge.mcp_server"],
      "env": {
        "SKILLFORGE_DB": "~/.skillforge/skillforge.db"
      }
    }
  }
}
```

---

## Advanced Patterns

### Skill Versioning

```python
# Register initial version
skill = forge.register_skill(
    name="code-formatter",
    tier1_metadata="Format code with black and isort",
    tier2_core="Format the given Python code using black...",
)

# Later: update with improved instructions
forge._registry.update_skill(skill.id, {
    "tier2_core": "Format the given Python code using black with line-length=88...",
})

# Bump version
forge._registry.version_skill(skill.id)
```

### Custom Routing Strategy

```python
from skillforge.core.loader import ProgressiveLoader

class CustomLoader(ProgressiveLoader):
    def load_skill(self, query, tier=1, routing="q_value", limit=5):
        skills = super().load_skill(query, tier=tier, routing=routing, limit=limit)
        # Apply custom filtering logic
        return [s for s in skills if self._meets_criteria(s)]
    
    def _meets_criteria(self, skill):
        return skill.q_value >= 0.3 and skill.usage_count >= 3
```

### Database Sharing Across Components

```python
# All components share the same SQLite database
from pathlib import Path
db = Path("~/.skillforge/shared.db").expanduser()

forge = SkillForge(db_path=db)
# The registry, tracker, and all components use the same DB
# This is the default behavior - forge._registry and forge._tracker
# both point to the same database file
```

### Multi-Agent Skill Sharing

```python
# Agent 1 registers skills
forge1 = SkillForge(db_path="/shared/skillforge.db")
forge1.register_skill("data-analysis", "Analyze datasets and produce insights")

# Agent 2 (different process) reads them
forge2 = SkillForge(db_path="/shared/skillforge.db")
skills = forge2.load_skill("analyze data")
# Agent 2 sees skills registered by Agent 1
```

---

## Next Steps

- See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for evaluating SkillForge performance
- See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for deep-dive into internals
