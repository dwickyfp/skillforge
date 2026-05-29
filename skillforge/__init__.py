"""
SkillForge: Self-Evolving Skill Intelligence Platform

A platform for making AI agent skills into living, evolving,
self-improving assets. Designed to work with Hermes Agent, OpenClaw,
and custom agent frameworks.

Core Components:
- SkillRegistry: 3-tier progressive loading
- EffectivenessTracker: Q-values via TD(lambda)
- ProgressiveLoader: Q-value routing with 4 strategies
- SkillDependencyGraph: BFS pathfinding, Q-propagation
- EvolutionLoop: Read-Execute-Reflect-Write pattern
- SelfDiagnosisEngine: AEL-inspired minimal self-diagnosis

Intelligence Layer:
- ConflictDetector: Overlap/contradiction detection
- HealthMonitor: Real-time skill health scoring
- SkillCreator: Auto-create from trajectories
- SkillAnalyzer: Clustering, patterns, recommendations
- SkillOptimizer: Compression, merging, reordering

API Layer:
- REST API server (http.server based)
- Python client library
- CLI tool

Based on converging research: Memento-Skills, AEL, SKILLREDUCER,
SEA-Eval, MemQ, SEARL.
"""

__version__ = "0.4.0"
__author__ = "Dwicky Feriansyah Putra"

from skillforge.forge import SkillForge
from skillforge.core.registry import SkillRegistry
from skillforge.core.tracker import QValueTracker, EffectivenessTracker
from skillforge.core.loader import ProgressiveLoader
from skillforge.core.graph import SkillDependencyGraph
from skillforge.core.evolution import EvolutionLoop
from skillforge.core.diagnosis import SelfDiagnosisEngine

__all__ = [
    "SkillForge",
    "SkillRegistry",
    "QValueTracker",
    "EffectivenessTracker",
    "ProgressiveLoader",
    "SkillDependencyGraph",
    "EvolutionLoop",
    "SelfDiagnosisEngine",
]
