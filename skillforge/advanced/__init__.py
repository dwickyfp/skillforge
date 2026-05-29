"""SkillForge Advanced — RL optimization, multi-agent, prediction, transfer, and elastic memory."""

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
]
