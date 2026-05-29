"""
Zero-shot skill generation from task descriptions.

Implements a template-driven skill generator that takes a natural-language
task description, infers structure (name, metadata, tags, core prompt),
and optionally evaluates and refines the generated skill through iterative
self-improvement.

The generator is fully stdlib-only — no LLM calls are made.  Instead it
uses heuristic NLP (regex tokenisation, TF-IDF-lite keyword extraction,
template assembly) to produce draft skills that can be registered in the
SkillRegistry and later refined by the RL optimizer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeneratedSkill:
    """A skill produced by :class:`SkillGenerator`.

    Attributes
    ----------
    skill_id : str
        Unique identifier for the generated skill.
    name : str
        Short human-readable name.
    tier1_metadata : str
        Compact routing metadata (~30 tokens).
    tier2_core : str
        Core prompt / instruction set.
    tier3_resources : list[str]
        Auxiliary resource references.
    tags : list[str]
        Inferred topic tags.
    confidence : float
        Generator confidence in ``[0, 1]``.
    rationale : str
        Short explanation of why this structure was chosen.
    """

    skill_id: str
    name: str
    tier1_metadata: str
    tier2_core: str
    tier3_resources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    rationale: str = ""


@dataclass
class GenerationRequest:
    """Input specification for skill generation.

    Parameters
    ----------
    description : str
        Natural-language description of the task the skill should solve.
    domain : str | None
        Optional domain hint (e.g. ``"python"``, ``"devops"``).
    complexity : str
        ``"simple"`` | ``"moderate"`` | ``"complex"``.
    max_tokens : int
        Soft cap on tier2_core token count.
    include_examples : bool
        Whether to generate example invocations.
    """

    description: str
    domain: str | None = None
    complexity: str = "moderate"
    max_tokens: int = 500
    include_examples: bool = True


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SkillRegistryProto:
    """Minimal registry interface for registering generated skills."""

    def register_skill(
        self,
        name: str,
        tier1_metadata: str,
        tier2_core: str = "",
        tier3_resources: list[str] | None = None,
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> Any: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Keyword extraction (TF-IDF-lite)
# ---------------------------------------------------------------------------

# Common English stop words — kept minimal to avoid external deps.
_STOP_WORDS: frozenset[str] = frozenset(
    "a an the and or but is are was were be been being have has had do does "
    "did will would shall should may might can could of in to for on with at "
    "by from as into through during before after above below between out off "
    "over under again further then once here there when where why how all both "
    "each few more most other some such no nor not only own same so than too "
    "very it its this that these those i me my we our you your he him his she "
    "her they them their what which who whom".split()
)


def _extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Extract top-N keywords from text using a TF-IDF-lite approach.

    Parameters
    ----------
    text : str
        Input text to extract keywords from.
    top_n : int
        Number of keywords to return.

    Returns
    -------
    list[str]
        Ranked list of keywords.
    """
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
    freq: dict[str, int] = {}
    for tok in tokens:
        if tok not in _STOP_WORDS:
            freq[tok] = freq.get(tok, 0) + 1
    # Score = frequency × length bonus (longer terms are often more specific)
    scored = sorted(freq.items(), key=lambda kv: kv[1] * (1 + len(kv[0]) * 0.1), reverse=True)
    return [kw for kw, _ in scored[:top_n]]


def _infer_tags(keywords: list[str], description: str, domain: str | None) -> list[str]:
    """Infer semantic tags from keywords, description, and optional domain.

    Parameters
    ----------
    keywords : list[str]
        Extracted keywords.
    description : str
        Original description text.
    domain : str | None
        Optional domain hint.

    Returns
    -------
    list[str]
        Deduplicated tags.
    """
    tags: list[str] = []
    if domain:
        tags.append(domain.lower().strip())

    # Domain heuristics from keywords
    _DOMAIN_SIGNALS: dict[str, list[str]] = {
        "python": ["python", "pip", "django", "flask", "pytest"],
        "javascript": ["javascript", "node", "npm", "react", "typescript", "vue"],
        "devops": ["docker", "kubernetes", "k8s", "ci", "cd", "terraform", "deploy"],
        "database": ["sql", "postgres", "mysql", "sqlite", "query", "database"],
        "security": ["auth", "oauth", "jwt", "encrypt", "hash", "security"],
        "data": ["pandas", "numpy", "dataframe", "csv", "etl", "spark"],
        "web": ["api", "rest", "graphql", "http", "endpoint", "webhook"],
        "testing": ["test", "mock", "assert", "coverage", "fixture"],
    }
    kw_set = set(keywords)
    desc_lower = description.lower()
    for tag, signals in _DOMAIN_SIGNALS.items():
        if tag in kw_set or any(s in kw_set or s in desc_lower for s in signals):
            tags.append(tag)

    # Add top 3 keywords as tags if not already present
    for kw in keywords[:3]:
        if kw not in tags and kw not in _STOP_WORDS:
            tags.append(kw)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        t_clean = t.strip().lower()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            unique.append(t_clean)
    return unique[:8]


# ---------------------------------------------------------------------------
# SkillGenerator
# ---------------------------------------------------------------------------


class SkillGenerator:
    """Zero-shot skill generator.

    Creates draft skills from natural-language descriptions using
    template-driven heuristic NLP — no external LLM dependency.

    Parameters
    ----------
    registry : SkillRegistryProto | None
        Optional registry for auto-registering generated skills.
    max_core_tokens : int
        Default soft cap on generated tier2_core token count.
    """

    def __init__(
        self,
        registry: SkillRegistryProto | None = None,
        max_core_tokens: int = 500,
    ) -> None:
        self._registry = registry
        self._max_core_tokens = max_core_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GeneratedSkill:
        """Generate a single skill from a :class:`GenerationRequest`.

        Parameters
        ----------
        request : GenerationRequest
            The generation specification.

        Returns
        -------
        GeneratedSkill
            The generated skill ready for registration.
        """
        desc = request.description.strip()
        if not desc:
            raise ValueError("Description must not be empty")

        keywords = _extract_keywords(desc, top_n=8)
        tags = _infer_tags(keywords, desc, request.domain)

        name = self._generate_name(desc, keywords)
        tier1 = self._generate_tier1(desc, keywords, tags)
        tier2 = self._generate_tier2(desc, keywords, tags, request)
        tier3 = self._generate_tier3(desc, keywords)
        confidence = self._estimate_confidence(desc, keywords)
        rationale = self._build_rationale(keywords, tags, request.complexity)

        skill = GeneratedSkill(
            skill_id=f"gen_{uuid.uuid4().hex[:12]}",
            name=name,
            tier1_metadata=tier1,
            tier2_core=tier2,
            tier3_resources=tier3,
            tags=tags,
            confidence=confidence,
            rationale=rationale,
        )
        logger.info(
            "Generated skill '%s' (confidence=%.2f, tags=%s)",
            skill.name,
            skill.confidence,
            skill.tags,
        )
        return skill

    def generate_and_register(
        self, request: GenerationRequest
    ) -> GeneratedSkill:
        """Generate a skill and register it in the linked registry.

        Parameters
        ----------
        request : GenerationRequest
            The generation specification.

        Returns
        -------
        GeneratedSkill
            The generated skill (now registered).

        Raises
        ------
        RuntimeError
            If no registry is configured.
        """
        if self._registry is None:
            raise RuntimeError(
                "No registry configured — use SkillGenerator(registry=…)"
            )
        skill = self.generate(request)
        self._registry.register_skill(
            name=skill.name,
            tier1_metadata=skill.tier1_metadata,
            tier2_core=skill.tier2_core,
            tier3_resources=skill.tier3_resources,
            tags=skill.tags,
            skill_id=skill.skill_id,
        )
        logger.info("Registered generated skill '%s'", skill.skill_id)
        return skill

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GeneratedSkill]:
        """Generate multiple skills in one call.

        Parameters
        ----------
        requests : list[GenerationRequest]
            List of generation requests.

        Returns
        -------
        list[GeneratedSkill]
            Generated skills in the same order as requests.
        """
        return [self.generate(r) for r in requests]

    def refine(
        self,
        skill: GeneratedSkill,
        feedback: str,
    ) -> GeneratedSkill:
        """Refine a generated skill using textual feedback.

        The feedback is analysed for additional keywords and structural
        hints, then merged into the existing skill content.

        Parameters
        ----------
        skill : GeneratedSkill
            The skill to refine.
        feedback : str
            Free-text feedback or correction.

        Returns
        -------
        GeneratedSkill
            A new :class:`GeneratedSkill` with updated content.
        """
        fb_keywords = _extract_keywords(feedback, top_n=5)
        existing_kw = set(_extract_keywords(skill.tier2_core, top_n=8))

        # Merge new keywords into tags
        existing_tag_set = set(skill.tags)
        new_tags = list(skill.tags)
        for kw in fb_keywords:
            if kw not in existing_tag_set and kw not in _STOP_WORDS:
                new_tags.append(kw)
                existing_tag_set.add(kw)
                new_tags = new_tags[:8]

        # Append feedback-derived guidance to core
        guidance_lines = self._feedback_to_guidance(feedback, fb_keywords)
        updated_core = skill.tier2_core
        if guidance_lines:
            updated_core += "\n\n## Refined guidance\n" + guidance_lines

        # Clamp token count
        max_tok = self._max_core_tokens
        updated_core = self._truncate_to_tokens(updated_core, max_tok)

        # Boost confidence slightly if feedback is positive
        positive_signals = {"good", "great", "correct", "perfect", "yes", "right"}
        boost = 0.05 if any(w in feedback.lower() for w in positive_signals) else 0.0

        return GeneratedSkill(
            skill_id=skill.skill_id,
            name=skill.name,
            tier1_metadata=skill.tier1_metadata,
            tier2_core=updated_core,
            tier3_resources=skill.tier3_resources,
            tags=new_tags,
            confidence=min(1.0, skill.confidence + boost),
            rationale=skill.rationale + f"\nRefined with: {feedback[:120]}",
        )

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    def _generate_name(self, desc: str, keywords: list[str]) -> str:
        """Generate a short skill name from the description and keywords.

        Takes the first 5 words of the description (capped at 60 chars),
        capitalised title-style.
        """
        words = desc.split()[:5]
        name = " ".join(words).strip(".,;:!?")
        # Capitalise first letter of each word
        name = " ".join(w.capitalize() for w in name.split())
        if len(name) > 60:
            name = name[:57] + "..."
        return name

    def _generate_tier1(
        self, desc: str, keywords: list[str], tags: list[str]
    ) -> str:
        """Generate ~30-token tier1 routing metadata.

        Combines a condensed description with top tags.
        """
        # Take first sentence
        first_sentence = re.split(r"[.!?]", desc)[0].strip()
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."
        tag_str = " ".join(f"#{t}" for t in tags[:4])
        tier1 = f"{first_sentence} | {tag_str}" if tag_str else first_sentence
        # Clamp to ~30 tokens
        return self._truncate_to_tokens(tier1, 30)

    def _generate_tier2(
        self,
        desc: str,
        keywords: list[str],
        tags: list[str],
        request: GenerationRequest,
    ) -> str:
        """Generate the core prompt / instruction set (tier2).

        Assembles a structured prompt from the description, keywords,
        and complexity level.
        """
        sections: list[str] = []

        # Header
        sections.append(f"# {self._generate_name(desc, keywords)}")
        sections.append("")

        # Objective
        sections.append("## Objective")
        sections.append(desc)
        sections.append("")

        # Steps (complexity-dependent)
        steps = self._generate_steps(desc, keywords, request.complexity)
        sections.append("## Steps")
        for idx, step in enumerate(steps, start=1):
            sections.append(f"{idx}. {step}")
        sections.append("")

        # Constraints
        constraints = self._generate_constraints(request.complexity)
        sections.append("## Constraints")
        for c in constraints:
            sections.append(f"- {c}")
        sections.append("")

        # Examples
        if request.include_examples:
            sections.append("## Example")
            sections.append(self._generate_example(desc, keywords))
            sections.append("")

        # Output format
        sections.append("## Output format")
        sections.append(self._infer_output_format(desc, keywords))

        core = "\n".join(sections)
        return self._truncate_to_tokens(core, request.max_tokens)

    def _generate_tier3(self, desc: str, keywords: list[str]) -> list[str]:
        """Generate placeholder tier3 resource references."""
        resources: list[str] = []
        # Infer a possible reference document
        if "api" in keywords or "endpoint" in keywords:
            resources.append("api_reference.md")
        if "test" in keywords or "testing" in keywords:
            resources.append("test_examples.py")
        if len(keywords) > 3:
            resources.append(f"glossary_{'_'.join(keywords[:3])}.md")
        return resources

    # ------------------------------------------------------------------
    # Step / constraint inference
    # ------------------------------------------------------------------

    def _generate_steps(
        self, desc: str, keywords: list[str], complexity: str
    ) -> list[str]:
        """Infer procedural steps from the description.

        Heuristic: parse imperative verbs and action phrases.
        """
        step_count = {"simple": 3, "moderate": 5, "complex": 8}.get(complexity, 5)

        # If the description already contains numbered steps, extract them
        numbered = re.findall(r"\d+[.)]\s*(.+?)(?=\d+[.)]|\Z)", desc, re.DOTALL)
        if numbered:
            return [s.strip().rstrip(". ") for s in numbered[:step_count]]

        # Otherwise, generate generic procedural steps from keywords
        action_verbs = [
            "Analyse",
            "Identify",
            "Implement",
            "Validate",
            "Refine",
            "Document",
            "Test",
            "Deploy",
            "Review",
            "Optimise",
        ]
        steps: list[str] = []
        for i in range(step_count):
            verb = action_verbs[i % len(action_verbs)]
            if i < len(keywords):
                steps.append(f"{verb} the {keywords[i]} component")
            else:
                steps.append(f"{verb} the solution for correctness")
        return steps

    def _generate_constraints(self, complexity: str) -> list[str]:
        """Generate complexity-appropriate constraints."""
        base = [
            "Use only standard library modules (no external dependencies)",
            "Include full type hints and docstrings",
            "Handle edge cases gracefully with informative error messages",
        ]
        if complexity in ("moderate", "complex"):
            base.extend([
                "Ensure idempotency — repeated runs should produce identical results",
                "Log key decisions at DEBUG level",
            ])
        if complexity == "complex":
            base.extend([
                "Design for extensibility — use Protocol-based interfaces",
                "Include performance considerations for large inputs",
            ])
        return base

    def _generate_example(self, desc: str, keywords: list[str]) -> str:
        """Generate a brief usage example."""
        kw_sample = keywords[0] if keywords else "input"
        return (
            f"Given the input related to '{kw_sample}', "
            f"the expected outcome is a well-structured result that "
            f"satisfies the described task requirements."
        )

    def _infer_output_format(self, desc: str, keywords: list[str]) -> str:
        """Infer likely output format from the description."""
        desc_lower = desc.lower()
        if "json" in desc_lower:
            return "Return a JSON object with the computed results."
        if "table" in desc_lower or "csv" in desc_lower:
            return "Return data in tabular format (list of dictionaries)."
        if "list" in desc_lower:
            return "Return an ordered list of results."
        if "report" in desc_lower:
            return "Return a structured text report with sections."
        return "Return a clear, structured result with appropriate formatting."

    # ------------------------------------------------------------------
    # Rationale builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rationale(
        keywords: list[str], tags: list[str], complexity: str
    ) -> str:
        """Build a human-readable rationale for the generation decision.

        Parameters
        ----------
        keywords : list[str]
            Extracted keywords.
        tags : list[str]
            Inferred tags.
        complexity : str
            Requested complexity level.

        Returns
        -------
        str
            Short rationale string.
        """
        parts: list[str] = [
            f"Extracted {len(keywords)} keywords: {', '.join(keywords[:5])}",
            f"Inferred {len(tags)} tags: {', '.join(tags[:4])}",
            f"Complexity: {complexity}",
        ]
        return ". ".join(parts) + "."

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    def _estimate_confidence(self, desc: str, keywords: list[str]) -> float:
        """Estimate generation confidence in ``[0.2, 0.95]``.

        Factors: description length, keyword count, structural signals.
        """
        score = 0.4  # baseline

        # Length bonus (longer descriptions provide more signal)
        word_count = len(desc.split())
        if word_count > 20:
            score += 0.1
        if word_count > 50:
            score += 0.1

        # Keyword richness
        if len(keywords) >= 5:
            score += 0.1
        if len(keywords) >= 8:
            score += 0.05

        # Structural signals (numbered lists, headings)
        if re.search(r"\d+[.)]", desc):
            score += 0.1
        if re.search(r"^#+\s", desc, re.MULTILINE):
            score += 0.1

        # Code blocks suggest well-defined task
        if "```" in desc:
            score += 0.05

        return max(0.2, min(0.95, score))

    # ------------------------------------------------------------------
    # Refinement helpers
    # ------------------------------------------------------------------

    def _feedback_to_guidance(
        self, feedback: str, keywords: list[str]
    ) -> str:
        """Convert feedback text into actionable guidance lines."""
        lines: list[str] = []
        sentences = re.split(r"[.!?]", feedback)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                lines.append(f"- {sent}")
        return "\n".join(lines[:5])

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using the word×4/3 heuristic."""
        if not text:
            return 0
        return int(len(text.split()) * 4 / 3)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximately *max_tokens* tokens.

        Uses the word×4/3 heuristic for estimation.
        """
        if not text:
            return text
        words = text.split()
        estimated = int(len(words) * 4 / 3)
        if estimated <= max_tokens:
            return text
        target_words = int(max_tokens * 3 / 4)
        return " ".join(words[:target_words])
