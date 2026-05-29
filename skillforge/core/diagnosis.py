"""
Self-Diagnosis Engine Module.

Analyzes skill failures, generates insights about root causes, and applies
automatic patches. Supports both LLM-assisted and rule-based analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """
    Represents a diagnostic insight generated from failure analysis.

    Attributes:
        root_cause: Description of the identified root cause.
        heuristic: The heuristic or rule that triggered this insight.
        patch_suggestion: Suggested fix or improvement.
        confidence: Confidence score (0.0 to 1.0).
        timestamp: When the insight was generated.
    """

    root_cause: str
    heuristic: str
    patch_suggestion: str
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize insight to a dictionary."""
        return {
            "root_cause": self.root_cause,
            "heuristic": self.heuristic,
            "patch_suggestion": self.patch_suggestion,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Insight":
        """Deserialize insight from a dictionary."""
        return cls(
            root_cause=data["root_cause"],
            heuristic=data["heuristic"],
            patch_suggestion=data["patch_suggestion"],
            confidence=data.get("confidence", 0.5),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now()
            ),
        )


class SkillRegistry(Protocol):
    """Protocol for skill registry interface."""

    def get_skill(self, skill_id: str) -> Optional[dict[str, Any]]:
        """Retrieve skill metadata by ID."""
        ...

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> bool:
        """Update skill metadata."""
        ...

    def list_skills(self) -> list[str]:
        """List all skill IDs."""
        ...


class FailureTracker(Protocol):
    """Protocol for failure tracking interface."""

    def get_failures(
        self, skill_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve recent failures for a skill."""
        ...

    def record_failure(
        self, skill_id: str, error: str, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Record a failure event."""
        ...


# Common error patterns for rule-based analysis
_ERROR_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern_regex, root_cause, heuristic_name, patch_suggestion)
    (
        r"(?i)timeout|timed?\s*out|deadline exceeded",
        "Execution timeout - skill takes too long to complete",
        "timeout_pattern",
        "Add timeout handling, break work into smaller chunks, or increase timeout threshold",
    ),
    (
        r"(?i)memory|out of memory|oom|memor[y]?\s*alloc",
        "Memory exhaustion - skill uses excessive memory",
        "memory_pattern",
        "Implement streaming/batching, reduce data held in memory, or add memory limits",
    ),
    (
        r"(?i)connection|refused|reset|network|unreachable|dns",
        "Network connectivity issue - external service unreachable",
        "network_pattern",
        "Add retry logic with exponential backoff, implement circuit breaker, or cache responses",
    ),
    (
        r"(?i)permission|access denied|forbidden|auth|unauthorized|403",
        "Permission/authentication failure - insufficient access rights",
        "auth_pattern",
        "Verify credentials, check token expiry, or update access permissions",
    ),
    (
        r"(?i)not found|missing|404|no such|does not exist",
        "Resource not found - required input or dependency missing",
        "missing_resource_pattern",
        "Add existence checks, provide defaults, or validate inputs before processing",
    ),
    (
        r"(?i)parse|decode|json|invalid format|syntax|malformed",
        "Input parsing failure - unexpected data format",
        "parsing_pattern",
        "Add input validation, implement fallback parsers, or sanitize inputs",
    ),
    (
        r"(?i)rate.?limit|throttl|too many|429",
        "Rate limiting - too many requests to external service",
        "rate_limit_pattern",
        "Implement request queuing, add delays between requests, or use caching",
    ),
    (
        r"(?i)null|none|nullptr|undefined|attribute\s*error|key\s*error",
        "Null/missing value - unexpected absence of required data",
        "null_value_pattern",
        "Add null checks, provide default values, or validate data presence",
    ),
    (
        r"(?i)recursion|stack overflow|depth",
        "Recursion depth exceeded - infinite or deep recursion",
        "recursion_pattern",
        "Convert to iterative approach, add depth limits, or use tail-call optimization",
    ),
    (
        r"(?i)type\s*error|cast|convert|incompatible type",
        "Type mismatch - incompatible data types in operation",
        "type_error_pattern",
        "Add type checking, implement type coercion, or validate input types",
    ),
]


class SelfDiagnosisEngine:
    """
    Engine for analyzing skill failures and generating actionable insights.

    Supports both LLM-assisted analysis (when an llm_fn is provided) and
    rule-based fallback analysis using pattern matching on error messages.

    Attributes:
        registry: Skill registry for accessing skill metadata.
        tracker: Failure tracker for retrieving failure history.
        llm_fn: Optional callable for LLM-assisted analysis.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        tracker: FailureTracker,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize the diagnosis engine.

        Args:
            registry: Skill registry instance.
            tracker: Failure tracker instance.
            llm_fn: Optional callable that takes a prompt string and returns
                    an LLM response string. Used for advanced analysis.
        """
        self._registry = registry
        self._tracker = tracker
        self._llm_fn = llm_fn

    def analyze_failures(
        self, skill_id: str, window: int = 10
    ) -> list[Insight]:
        """
        Analyze recent failures for a skill and generate insights.

        Retrieves the most recent failures and analyzes them for patterns.
        Uses LLM analysis if available, otherwise falls back to rule-based
        pattern matching.

        Args:
            skill_id: The skill to analyze.
            window: Number of recent failures to consider. Default 10.

        Returns:
            List of Insight objects describing identified issues.

        Raises:
            ValueError: If skill_id is empty.
        """
        if not skill_id:
            raise ValueError("skill_id must be a non-empty string")

        failures = self._tracker.get_failures(skill_id, limit=window)

        if not failures:
            logger.debug("No failures found for skill '%s'", skill_id)
            return []

        if self._llm_fn:
            return self._llm_analysis(skill_id, failures)
        else:
            return self._rule_based_analysis(failures)

    def generate_insight(self, failures: list[dict[str, Any]]) -> Insight:
        """
        Generate a single consolidated insight from a list of failures.

        Args:
            failures: List of failure dictionaries with 'error' and optional
                      'context' keys.

        Returns:
            A consolidated Insight object.

        Raises:
            ValueError: If failures list is empty.
        """
        if not failures:
            raise ValueError("Cannot generate insight from empty failures list")

        if self._llm_fn:
            return self._llm_generate_insight(failures)
        else:
            return self._rule_based_generate_insight(failures)

    def auto_patch_skill(
        self, skill_id: str, insight: Insight
    ) -> dict[str, Any]:
        """
        Apply an automatic patch to a skill based on a diagnostic insight.

        Records the patch suggestion in the skill's metadata and updates
        its status to indicate a patch has been applied.

        Args:
            skill_id: The skill to patch.
            insight: The insight containing the patch suggestion.

        Returns:
            Dictionary with patch result information including success status.
        """
        skill = self._registry.get_skill(skill_id)
        if skill is None:
            return {
                "success": False,
                "skill_id": skill_id,
                "error": f"Skill '{skill_id}' not found in registry",
            }

        try:
            patch_record = {
                "applied_at": datetime.now().isoformat(),
                "root_cause": insight.root_cause,
                "heuristic": insight.heuristic,
                "patch_suggestion": insight.patch_suggestion,
                "confidence": insight.confidence,
            }

            # Get existing patches or initialize
            existing_patches = getattr(skill, "patches", []) or []
            existing_patches.append(patch_record)

            updates = {
                "patches": existing_patches,
                "last_patched": datetime.now().isoformat(),
                "patch_count": len(existing_patches),
            }

            success = self._registry.update_skill(skill_id, updates)

            if success:
                logger.info(
                    "Applied patch to skill '%s': %s",
                    skill_id,
                    insight.patch_suggestion,
                )
                return {
                    "success": True,
                    "skill_id": skill_id,
                    "patch": patch_record,
                    "total_patches": len(existing_patches),
                }
            else:
                return {
                    "success": False,
                    "skill_id": skill_id,
                    "error": "Registry update failed",
                }

        except Exception as e:
            logger.error(
                "Failed to auto-patch skill '%s': %s", skill_id, str(e)
            )
            return {
                "success": False,
                "skill_id": skill_id,
                "error": str(e),
            }

    def _rule_based_analysis(self, failures: list[dict[str, Any]]) -> list[Insight]:
        """
        Analyze failures using rule-based pattern matching.

        Args:
            failures: List of failure dictionaries.

        Returns:
            List of Insights from matched patterns.
        """
        insights: list[Insight] = []
        error_messages = [f.get("error", "") for f in failures]
        combined_text = " ".join(error_messages)

        matched_patterns: set[str] = set()

        for pattern_regex, root_cause, heuristic, suggestion in _ERROR_PATTERNS:
            match_count = sum(
                1 for msg in error_messages if re.search(pattern_regex, msg)
            )
            if match_count > 0 and heuristic not in matched_patterns:
                matched_patterns.add(heuristic)
                confidence = min(1.0, match_count / len(failures))
                insights.append(
                    Insight(
                        root_cause=root_cause,
                        heuristic=heuristic,
                        patch_suggestion=suggestion,
                        confidence=confidence,
                    )
                )

        # If no patterns matched, generate a generic insight
        if not insights:
            insights.append(
                Insight(
                    root_cause="Unclassified errors - no known pattern matched",
                    heuristic="fallback_generic",
                    patch_suggestion=(
                        "Review error logs manually. Consider adding structured "
                        "error reporting for better pattern detection."
                    ),
                    confidence=0.2,
                )
            )

        # Check for frequency-based issues
        if len(failures) >= 5:
            timestamps = [
                datetime.fromisoformat(f["timestamp"])
                for f in failures
                if "timestamp" in f
            ]
            if len(timestamps) >= 2:
                timestamps.sort()
                time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                if time_span > 0:
                    failure_rate = len(failures) / (time_span / 3600)  # per hour
                    if failure_rate > 10:
                        insights.append(
                            Insight(
                                root_cause=(
                                    f"High failure rate: {failure_rate:.1f} failures/hour. "
                                    "Indicates systemic issue rather than transient errors."
                                ),
                                heuristic="high_frequency",
                                patch_suggestion=(
                                    "Consider degrading gracefully, implementing circuit "
                                    "breaker, or temporarily disabling the skill."
                                ),
                                confidence=0.8,
                            )
                        )

        return insights

    def _rule_based_generate_insight(
        self, failures: list[dict[str, Any]]
    ) -> Insight:
        """
        Generate a single consolidated insight using rule-based analysis.

        Args:
            failures: List of failure dictionaries.

        Returns:
            A single consolidated Insight.
        """
        insights = self._rule_based_analysis(failures)

        if not insights:
            return Insight(
                root_cause="No failures to analyze",
                heuristic="none",
                patch_suggestion="No action needed",
                confidence=1.0,
            )

        # Return the highest confidence insight, enriched with count info
        best = max(insights, key=lambda i: i.confidence)
        best.root_cause = (
            f"[{len(failures)} failures analyzed] {best.root_cause}"
        )
        return best

    def _llm_analysis(
        self, skill_id: str, failures: list[dict[str, Any]]
    ) -> list[Insight]:
        """
        Analyze failures using LLM for advanced pattern recognition.

        Falls back to rule-based analysis if the LLM call fails.

        Args:
            skill_id: The skill being analyzed.
            failures: List of failure dictionaries.

        Returns:
            List of Insights from LLM analysis.
        """
        prompt = self._build_analysis_prompt(skill_id, failures)

        try:
            response = self._llm_fn(prompt)  # type: ignore[misc]
            insights = self._parse_llm_response(response)
            if insights:
                return insights
        except Exception as e:
            logger.warning(
                "LLM analysis failed for skill '%s', falling back to rules: %s",
                skill_id,
                str(e),
            )

        # Fallback to rule-based
        return self._rule_based_analysis(failures)

    def _llm_generate_insight(self, failures: list[dict[str, Any]]) -> Insight:
        """
        Generate a single insight using LLM analysis.

        Falls back to rule-based if LLM fails.

        Args:
            failures: List of failure dictionaries.

        Returns:
            A single consolidated Insight.
        """
        prompt = (
            "Analyze the following skill failures and provide a single consolidated insight.\n"
            "Respond in this exact format:\n"
            "ROOT_CAUSE: <description>\n"
            "HEURISTIC: <name>\n"
            "PATCH: <suggestion>\n"
            "CONFIDENCE: <0.0-1.0>\n\n"
            f"Failures:\n{self._format_failures(failures)}"
        )

        try:
            response = self._llm_fn(prompt)  # type: ignore[misc]
            insights = self._parse_llm_response(response)
            if insights:
                return insights[0]
        except Exception as e:
            logger.warning("LLM insight generation failed, falling back: %s", str(e))

        return self._rule_based_generate_insight(failures)

    def _build_analysis_prompt(
        self, skill_id: str, failures: list[dict[str, Any]]
    ) -> str:
        """Build the prompt for LLM-based analysis."""
        return (
            f"You are a skill failure analyst. Analyze these failures for skill '{skill_id}'.\n\n"
            f"Failures ({len(failures)} total):\n"
            f"{self._format_failures(failures)}\n\n"
            "For each distinct issue pattern found, provide:\n"
            "ROOT_CAUSE: <description>\n"
            "HEURISTIC: <pattern_name>\n"
            "PATCH: <actionable suggestion>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "---\n"
            "Identify all distinct failure patterns."
        )

    def _format_failures(self, failures: list[dict[str, Any]]) -> str:
        """Format failures into a readable string for prompts."""
        lines: list[str] = []
        for i, f in enumerate(failures, 1):
            error = f.get("error", "unknown")
            timestamp = f.get("timestamp", "N/A")
            context = f.get("context", {})
            lines.append(f"  {i}. [{timestamp}] {error}")
            if context:
                lines.append(f"     Context: {context}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str) -> list[Insight]:
        """
        Parse structured LLM response into Insight objects.

        Args:
            response: Raw LLM response text.

        Returns:
            List of parsed Insights (empty if parsing fails).
        """
        insights: list[Insight] = []

        # Split by separator or by ROOT_CAUSE markers
        blocks = re.split(r"(?:^|\n)---\s*(?:\n|$)", response)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            root_cause_match = re.search(
                r"ROOT_CAUSE:\s*(.+?)(?=\nHEURISTIC:|\nPATCH:|\nCONFIDENCE:|\Z)",
                block,
                re.DOTALL,
            )
            heuristic_match = re.search(
                r"HEURISTIC:\s*(.+?)(?=\nROOT_CAUSE:|\nPATCH:|\nCONFIDENCE:|\Z)",
                block,
                re.DOTALL,
            )
            patch_match = re.search(
                r"PATCH:\s*(.+?)(?=\nROOT_CAUSE:|\nHEURISTIC:|\nCONFIDENCE:|\Z)",
                block,
                re.DOTALL,
            )
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", block)

            if root_cause_match and patch_match:
                insights.append(
                    Insight(
                        root_cause=root_cause_match.group(1).strip(),
                        heuristic=(
                            heuristic_match.group(1).strip()
                            if heuristic_match
                            else "llm_analysis"
                        ),
                        patch_suggestion=patch_match.group(1).strip(),
                        confidence=(
                            min(1.0, max(0.0, float(confidence_match.group(1))))
                            if confidence_match
                            else 0.6
                        ),
                    )
                )

        return insights
