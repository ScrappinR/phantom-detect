"""
Statistical Anomaly Detection

Detects covert channel activity through statistical analysis.
Looks for deviations from baseline behavior that might indicate
intentional encoding in LLM outputs.

Detection metrics:
    - Entropy deviation: Token distribution entropy varies from baseline
    - Timing deviation: Inter-action timing varies from baseline
    - Pattern deviation: Tool/structure patterns deviate from expected
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .types import ChannelType, DetectionAlert, DetectionSeverity
from .baseline import Baseline, BaselineManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ccds import AgentActivity


@dataclass
class AnomalyScore:
    """
    Score for a potential anomaly.

    Combines multiple detection metrics into an overall score.
    """

    overall_score: float  # 0-1, higher = more anomalous
    entropy_score: float
    timing_score: float
    pattern_score: float
    contributing_factors: List[str]

    @property
    def is_anomalous(self) -> bool:
        """Check if score exceeds anomaly threshold."""
        return self.overall_score > 0.5


class StatisticalDetector:
    """
    Statistical anomaly detector for covert channels in LLM outputs.

    Analyzes agent behavior for statistical deviations that
    might indicate covert channel encoding.

    Example:
        detector = StatisticalDetector(baseline_manager)
        score = detector.analyze(activity, baseline)
        if score.is_anomalous:
            alert = detector.create_alert(activity, score)
    """

    def __init__(
        self,
        baseline_manager: Optional[BaselineManager] = None,
        sensitivity: float = 0.7
    ):
        """
        Initialize the statistical detector.

        Args:
            baseline_manager: Manager for baseline data
            sensitivity: Detection sensitivity (0-1), higher = more alerts
        """
        self._baseline_manager = baseline_manager or BaselineManager()
        self._sensitivity = sensitivity

        # Z-score thresholds (adjusted by sensitivity)
        self._base_threshold = 3.0
        self._threshold = self._base_threshold - (sensitivity * 2)

    def analyze(
        self,
        activity: 'AgentActivity',
        baseline: Optional[Baseline] = None
    ) -> AnomalyScore:
        """
        Analyze activity for statistical anomalies.

        Args:
            activity: The activity to analyze
            baseline: Baseline to compare against (fetched if None)

        Returns:
            AnomalyScore with detection metrics
        """
        if baseline is None:
            baseline = self._baseline_manager.get_baseline(activity.agent_id)

        factors = []

        entropy_score = self._analyze_entropy(activity, baseline)
        if entropy_score > 0.5:
            factors.append(f"entropy_deviation({entropy_score:.2f})")

        timing_score = self._analyze_timing(activity, baseline)
        if timing_score > 0.5:
            factors.append(f"timing_deviation({timing_score:.2f})")

        pattern_score = self._analyze_patterns(activity, baseline)
        if pattern_score > 0.5:
            factors.append(f"pattern_deviation({pattern_score:.2f})")

        overall = self._combine_scores(entropy_score, timing_score, pattern_score)

        return AnomalyScore(
            overall_score=overall,
            entropy_score=entropy_score,
            timing_score=timing_score,
            pattern_score=pattern_score,
            contributing_factors=factors
        )

    def _analyze_entropy(
        self,
        activity: 'AgentActivity',
        baseline: Baseline
    ) -> float:
        """Analyze entropy deviation from baseline. Returns score 0-1."""
        entropy = activity.content.get("entropy")

        if entropy is None:
            return 0.0

        z_score = baseline.z_score("entropy", entropy)
        score = z_score / (2 * self._threshold)
        return min(1.0, max(0.0, score))

    def _analyze_timing(
        self,
        activity: 'AgentActivity',
        baseline: Baseline
    ) -> float:
        """Analyze timing deviation from baseline."""
        if activity.metadata is None:
            return 0.0

        timing = activity.metadata.get("time_since_last_ms")
        if timing is None:
            return 0.0

        z_score = baseline.z_score("timing", timing)
        score = z_score / (2 * self._threshold)
        return min(1.0, max(0.0, score))

    def _analyze_patterns(
        self,
        activity: 'AgentActivity',
        baseline: Baseline
    ) -> float:
        """Analyze pattern deviation from baseline."""
        score = 0.0
        count = 0

        if activity.activity_type == "tool_call":
            tool_score = self._analyze_tool_pattern(activity, baseline)
            score += tool_score
            count += 1

        if activity.activity_type == "generation":
            struct_score = self._analyze_structure_pattern(activity, baseline)
            score += struct_score
            count += 1

        if count == 0:
            return 0.0

        return score / count

    def _analyze_tool_pattern(
        self,
        activity: 'AgentActivity',
        baseline: Baseline
    ) -> float:
        """Analyze tool call pattern deviation."""
        tool_calls = activity.content.get("tool_calls", [])

        if len(tool_calls) < 2:
            return 0.0

        tools = []
        for call in tool_calls[-2:]:
            if isinstance(call, dict):
                name = call.get("tool_name") or call.get("name")
                if name:
                    tools.append(name)

        if len(tools) < 2:
            return 0.0

        pattern = f"{tools[0]}->{tools[1]}"
        expected_freq = baseline.tool_call_patterns.get(pattern, 0.0)

        if expected_freq < 0.05 and baseline.sample_count > 50:
            return 0.7
        elif expected_freq < 0.1:
            return 0.4
        else:
            return 0.1

    def _analyze_structure_pattern(
        self,
        activity: 'AgentActivity',
        baseline: Baseline
    ) -> float:
        """Analyze structure pattern deviation."""
        text = activity.content.get("text", "")

        if not text:
            return 0.0

        if text.strip().startswith(("1.", "- ", "* ")):
            pattern = "enumerated"
        elif "?" in text:
            pattern = "questioning"
        else:
            pattern = "prose"

        expected_freq = baseline.structure_patterns.get(pattern, 0.33)

        if expected_freq < 0.1 and baseline.sample_count > 20:
            return 0.6
        elif expected_freq < 0.2:
            return 0.3
        else:
            return 0.1

    def _combine_scores(
        self,
        entropy_score: float,
        timing_score: float,
        pattern_score: float
    ) -> float:
        """Combine individual scores into overall anomaly score."""
        weights = {
            "entropy": 0.4,
            "timing": 0.3,
            "pattern": 0.3
        }

        weighted_sum = (
            weights["entropy"] * entropy_score +
            weights["timing"] * timing_score +
            weights["pattern"] * pattern_score
        )

        adjusted = weighted_sum * (0.5 + self._sensitivity * 0.5)
        return min(1.0, adjusted)

    def create_alert(
        self,
        activity: 'AgentActivity',
        score: AnomalyScore
    ) -> DetectionAlert:
        """
        Create detection alert from anomaly score.

        Args:
            activity: The triggering activity
            score: The anomaly score

        Returns:
            DetectionAlert for the detected anomaly
        """
        if score.overall_score > 0.8:
            severity = DetectionSeverity.HIGH
        elif score.overall_score > 0.6:
            severity = DetectionSeverity.MEDIUM
        elif score.overall_score > 0.4:
            severity = DetectionSeverity.LOW
        else:
            severity = DetectionSeverity.INFO

        channel_type = None
        if "entropy" in str(score.contributing_factors):
            channel_type = ChannelType.EGE
        elif "pattern" in str(score.contributing_factors):
            if activity.activity_type == "tool_call":
                channel_type = ChannelType.TCOE
            else:
                channel_type = ChannelType.COTSE
        elif "timing" in str(score.contributing_factors):
            channel_type = ChannelType.HSCE

        return DetectionAlert(
            severity=severity,
            channel_type=channel_type,
            description=f"Statistical anomaly detected (score: {score.overall_score:.2f})",
            evidence={
                "overall_score": score.overall_score,
                "entropy_score": score.entropy_score,
                "timing_score": score.timing_score,
                "pattern_score": score.pattern_score,
                "factors": score.contributing_factors
            },
            source_agent_id=activity.agent_id,
            recommended_action="Review agent output for potential covert encoding"
        )
