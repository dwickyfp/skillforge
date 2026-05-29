# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-05-29

### Added — Advanced Layer
- **RLOptimizer** — Reinforcement-learning inspired skill optimizer with iterative compression, split, and reorder actions; composite reward signal; training step logging
- **SharedSkillPool** — Multi-agent skill sharing with isolated workspaces, fine-grained access control (read/write/admin), and cross-workspace Q-value sync
- **SkillPredictor** — Skill performance predictor using OLS linear regression on historical outcomes; decline detection; context-aware skill recommendations
- **SkillTransferEngine** — Cross-agent skill transfer supporting Hermes (SKILL.md), OpenClaw (.learnings/), JSON, and Markdown formats; bidirectional import/export; batch transfer
- **MCP Server** — Model Context Protocol server exposing SkillForge tools over stdin/stdout JSON-RPC for agent interop
- Entry points: `skillforge` CLI and `skillforge-mcp` server commands

## [0.3.0] - 2026-05-29

### Added — Platform Layer
- **REST API** — Full HTTP API server (stdlib `http.server`) with endpoints for skill CRUD, search, evolution, health, and dashboard
- **Web Dashboard** — HTML dashboard for real-time skill health visualization
- **CLI** — Unified command-line interface with subcommands for skills, server, evolution, health, dashboard, and import/export
- **API Client** — Programmatic Python client for the REST API

## [0.2.0] - 2026-05-29

### Added — Intelligence Layer
- **ConflictDetector** — Detects overlapping skills, contradictory instructions, and circular dependencies; supports merge, deprecate, and manual resolution strategies
- **HealthMonitor** — Real-time skill health scoring (0–1) based on Q-value, success rate, recency, and usage; classifies into healthy/warning/critical/dead buckets; dashboard summary
- **SkillCreator** — Auto-creates skills from agent execution trajectories; extracts common steps; identifies failure pitfalls; optional LLM-enhanced generation
- **SkillAnalyzer** — Advanced analytics with domain/Q-value clustering, usage pattern analysis, and temporal trend detection
- **SkillOptimizer** — Structural optimization via compression (dedup, filler removal), splitting, merging, and step reordering

## [0.1.0] - 2026-05-29

### Added
- Initial release with all 6 core components:
  - **SkillRegistry** — skill discovery, versioning, and dependency resolution
  - **ExecutionEngine** — sandboxed skill execution with lifecycle management
  - **SkillEvolver** — evolutionary optimization and mutation of skills
  - **SkillComposer** — pipeline composition and skill chaining
  - **SkillBenchmark** — performance benchmarking and evaluation
  - **SkillVisualizer** — metrics visualization and reporting
