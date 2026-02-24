"""
Covert Channel Detection System (CCDS)

Main detection engine for phantom-detect.
Monitors agent activity and detects potential covert channel usage
in LLM outputs through statistical anomaly detection, cross-agent
correlation, and pattern recognition.

Detection approaches:
    1. Statistical Anomaly Detection - deviation from baseline behavior
    2. Cross-Agent Correlation - synchronized behavior patterns
    3. Pattern Recognition - known encoding signatures
    4. Timing Analysis - suspicious timing patterns
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import logging

from .types import ChannelType, DetectionAlert, DetectionSeverity
from .baseline import Baseline, BaselineManager
from .statistical import StatisticalDetector, AnomalyScore


@dataclass
class AgentActivity:
    """
    Container for agent activity to analyze.

    Represents a single action or output from an AI agent
    that should be analyzed for covert channel indicators.

    Attributes:
        agent_id: Which agent generated this activity
        timestamp: When the activity occurred
        activity_type: Type of activity ("generation", "tool_call", etc.)
        content: Activity-specific content (must include relevant fields
                 like "entropy", "text", "tool_calls")
        metadata: Additional context (e.g. "time_since_last_ms")
    """

    agent_id: str
    timestamp: datetime
    activity_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CCDS:
    """
    Covert Channel Detection System.

    Monitors agent activity for indicators of covert channel usage.
    Designed to detect steganographic encoding in LLM outputs,
    tool call patterns, and multi-agent coordination.

    Example:
        ccds = CCDS(sensitivity=0.7)

        # Analyze activity
        alerts = ccds.analyze_activity(activity)
        for alert in alerts:
            if alert.severity >= DetectionSeverity.MEDIUM:
                handle_alert(alert)

        # Register custom alert handler
        ccds.register_alert_handler(my_handler)
    """

    def __init__(
        self,
        sensitivity: float = 0.7,
        baseline_window: timedelta = timedelta(hours=24),
        correlation_window: timedelta = timedelta(seconds=5),
        enable_cross_agent: bool = True
    ):
        """
        Initialize CCDS.

        Args:
            sensitivity: Detection sensitivity (0-1). Higher = more alerts.
            baseline_window: Time window for baseline calculation.
            correlation_window: Window for cross-agent correlation.
            enable_cross_agent: Whether to check cross-agent correlation.
        """
        self._sensitivity = sensitivity
        self._correlation_window = correlation_window
        self._enable_cross_agent = enable_cross_agent

        self._baseline_manager = BaselineManager(window_size=baseline_window)
        self._statistical_detector = StatisticalDetector(
            baseline_manager=self._baseline_manager,
            sensitivity=sensitivity
        )

        self._recent_activities: Dict[str, List[AgentActivity]] = defaultdict(list)
        self._max_recent = 100

        self._alert_handlers: List[Callable[[DetectionAlert], None]] = []
        self._alert_history: List[DetectionAlert] = []
        self._max_alert_history = 1000

        self._logger = logging.getLogger("phantom_detect.ccds")

    def analyze_activity(
        self,
        activity: AgentActivity
    ) -> List[DetectionAlert]:
        """
        Analyze an activity for covert channel indicators.

        Performs statistical and correlation checks and returns any alerts.

        Args:
            activity: The activity to analyze

        Returns:
            List of detection alerts (may be empty)
        """
        alerts = []

        self._record_activity(activity)

        baseline = self._baseline_manager.get_baseline(activity.agent_id)

        stat_alerts = self._run_statistical_detection(activity, baseline)
        alerts.extend(stat_alerts)

        if self._enable_cross_agent:
            corr_alerts = self._run_correlation_detection(activity)
            alerts.extend(corr_alerts)

        for alert in alerts:
            self._record_alert(alert)
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self._logger.error(f"Alert handler error: {e}")

        return alerts

    def _record_activity(self, activity: AgentActivity):
        """Record activity for analysis."""
        self._recent_activities[activity.agent_id].append(activity)

        if len(self._recent_activities[activity.agent_id]) > self._max_recent:
            self._recent_activities[activity.agent_id] = (
                self._recent_activities[activity.agent_id][-self._max_recent:]
            )

        self._baseline_manager.record_activity(activity)

    def _run_statistical_detection(
        self,
        activity: AgentActivity,
        baseline: Baseline
    ) -> List[DetectionAlert]:
        """Run statistical anomaly detection."""
        alerts = []

        score = self._statistical_detector.analyze(activity, baseline)

        if score.is_anomalous:
            alert = self._statistical_detector.create_alert(activity, score)
            alerts.append(alert)

            self._logger.info(
                f"Statistical anomaly detected for {activity.agent_id}: "
                f"score={score.overall_score:.2f}"
            )

        return alerts

    def _run_correlation_detection(
        self,
        activity: AgentActivity
    ) -> List[DetectionAlert]:
        """
        Detect correlated behavior between agents.

        Looks for near-simultaneous activities that might indicate
        synchronized covert communication.
        """
        alerts = []
        now = activity.timestamp

        correlated_agents = []

        for agent_id, activities in self._recent_activities.items():
            if agent_id == activity.agent_id:
                continue

            for other in activities[-10:]:
                time_diff = abs((other.timestamp - now).total_seconds())
                if time_diff < self._correlation_window.total_seconds():
                    correlated_agents.append({
                        "agent_id": agent_id,
                        "time_diff": time_diff,
                        "activity_type": other.activity_type
                    })
                    break

        if len(correlated_agents) >= 2:
            alert = DetectionAlert(
                severity=DetectionSeverity.MEDIUM,
                channel_type=ChannelType.CASP,
                description=(
                    f"Correlated activity detected across "
                    f"{len(correlated_agents) + 1} agents"
                ),
                evidence={
                    "source_agent": activity.agent_id,
                    "correlated_agents": correlated_agents,
                    "window_seconds": self._correlation_window.total_seconds()
                },
                source_agent_id=activity.agent_id,
                recommended_action=(
                    "Investigate potential cross-agent coordination"
                )
            )
            alerts.append(alert)

            self._logger.warning(
                f"Cross-agent correlation detected: "
                f"{activity.agent_id} + {len(correlated_agents)} others"
            )

        return alerts

    def _record_alert(self, alert: DetectionAlert):
        """Record alert in history."""
        self._alert_history.append(alert)

        if len(self._alert_history) > self._max_alert_history:
            self._alert_history = self._alert_history[-self._max_alert_history:]

    def register_alert_handler(
        self,
        handler: Callable[[DetectionAlert], None]
    ):
        """
        Register a handler for detection alerts.

        Args:
            handler: Callable that takes a DetectionAlert
        """
        self._alert_handlers.append(handler)

    def unregister_alert_handler(
        self,
        handler: Callable[[DetectionAlert], None]
    ):
        """Unregister an alert handler."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def get_baseline(self, agent_id: str) -> Baseline:
        """Get baseline for an agent."""
        return self._baseline_manager.get_baseline(agent_id)

    def get_recent_alerts(
        self,
        since: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        min_severity: DetectionSeverity = DetectionSeverity.INFO
    ) -> List[DetectionAlert]:
        """
        Get recent alerts matching criteria.

        Args:
            since: Only alerts after this time
            agent_id: Only alerts for this agent
            min_severity: Minimum severity level

        Returns:
            List of matching alerts
        """
        alerts = self._alert_history

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if agent_id:
            alerts = [a for a in alerts if a.source_agent_id == agent_id]

        alerts = [a for a in alerts if a.severity.value >= min_severity.value]

        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about detection alerts."""
        if not self._alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_channel": {},
                "by_agent": {}
            }

        by_severity: Dict[str, int] = defaultdict(int)
        by_channel: Dict[str, int] = defaultdict(int)
        by_agent: Dict[str, int] = defaultdict(int)

        for alert in self._alert_history:
            by_severity[alert.severity.name] += 1
            if alert.channel_type:
                by_channel[alert.channel_type.name] += 1
            if alert.source_agent_id:
                by_agent[alert.source_agent_id] += 1

        return {
            "total_alerts": len(self._alert_history),
            "by_severity": dict(by_severity),
            "by_channel": dict(by_channel),
            "by_agent": dict(by_agent),
            "oldest_alert": self._alert_history[0].timestamp.isoformat(),
            "newest_alert": self._alert_history[-1].timestamp.isoformat()
        }

    def clear_history(self, agent_id: Optional[str] = None):
        """
        Clear activity and alert history.

        Args:
            agent_id: Clear only for this agent (None = all)
        """
        if agent_id:
            if agent_id in self._recent_activities:
                del self._recent_activities[agent_id]
            self._alert_history = [
                a for a in self._alert_history
                if a.source_agent_id != agent_id
            ]
            self._baseline_manager.clear_history(agent_id)
        else:
            self._recent_activities.clear()
            self._alert_history.clear()

    def set_sensitivity(self, sensitivity: float):
        """
        Adjust detection sensitivity.

        Args:
            sensitivity: New sensitivity (0-1)
        """
        self._sensitivity = max(0.0, min(1.0, sensitivity))
        self._statistical_detector = StatisticalDetector(
            baseline_manager=self._baseline_manager,
            sensitivity=self._sensitivity
        )

    @property
    def sensitivity(self) -> float:
        """Current detection sensitivity."""
        return self._sensitivity

    def __repr__(self) -> str:
        return (
            f"<CCDS sensitivity={self._sensitivity:.2f} "
            f"agents={len(self._recent_activities)} "
            f"alerts={len(self._alert_history)}>"
        )
