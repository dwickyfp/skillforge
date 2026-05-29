"""SkillForge Advanced — RL optimization, multi-agent, prediction, transfer, elastic memory, and A/B testing."""

from skillforge.advanced.rl_optimizer import (
    RLOptimizer,
    TrainingStep,
    Experience,
    ReplayBuffer,
    ContextualBandit,
    RewardModel,
    CurriculumScheduler,
)
from skillforge.advanced.multi_agent import SharedSkillPool, AgentAccess, AccessLevel
from skillforge.advanced.predictor import SkillPredictor, SkillPrediction
from skillforge.advanced.transfer import SkillTransferEngine, TransferResult
from skillforge.advanced.skill_generator import (
    SkillGenerator,
    GeneratedSkill,
    GenerationRequest,
)
from skillforge.advanced.elastic_memory import ElasticMemory, MemoryEntry
from skillforge.advanced.ab_testing import (
    ABTestRunner,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    Variant,
    Outcome,
    VariantStats,
    MetricType,
    AssignmentStrategy,
)

__all__ = [
    "RLOptimizer",
    "TrainingStep",
    "Experience",
    "ReplayBuffer",
    "ContextualBandit",
    "RewardModel",
    "CurriculumScheduler",
    "SharedSkillPool",
    "AgentAccess",
    "AccessLevel",
    "SkillPredictor",
    "SkillPrediction",
    "SkillTransferEngine",
    "TransferResult",
    "SkillGenerator",
    "GeneratedSkill",
    "GenerationRequest",
    "ElasticMemory",
    "MemoryEntry",
    "ABTestRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "Variant",
    "Outcome",
    "VariantStats",
    "MetricType",
    "AssignmentStrategy",
]
