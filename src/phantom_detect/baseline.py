"""
Baseline Behavior Modeling

Maintains statistical baselines of agent behavior for anomaly detection.
Baselines are agent-specific and context-aware.

You can't spot unusual behavior without knowing what usual behavior looks like.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ccds import AgentActivity


@dataclass
class Baseline:
    """
    Statistical baseline for an agent's behavior.

    Captures normal ranges for various metrics to enable
    anomaly detection.

    Attributes:
        agent_id: Which agent this baseline describes
        entropy_mean: Mean entropy of token distributions
        entropy_std: Standard deviation of entropy
        tool_call_patterns: Frequency of tool call sequences
        structure_patterns: Frequency of structure types
        timing_mean: Mean inter-action timing (ms)
        timing_std: Standard deviation of timing
        sample_count: Number of samples in baseline
        last_updated: When baseline was last computed
    """

    agent_id: str
    entropy_mean: float = 2.5
    entropy_std: float = 0.5
    tool_call_patterns: Dict[str, float] = field(default_factory=dict)
    structure_patterns: Dict[str, float] = field(default_factory=dict)
    timing_mean: float = 1000.0
    timing_std: float = 500.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_stale(self, max_age: timedelta = timedelta(hours=24)) -> bool:
        """Check if baseline needs refresh."""
        return datetime.now(timezone.utc) - self.last_updated > max_age

    def z_score(self, metric: str, value: float) -> float:
        """
        Calculate z-score for a metric value.

        Args:
            metric: Name of metric ("entropy", "timing")
            value: Observed value

        Returns:
            Z-score (standard deviations from mean)
        """
        if metric == "entropy":
            if self.entropy_std == 0:
                return 0.0
            return abs(value - self.entropy_mean) / self.entropy_std
        elif metric == "timing":
            if self.timing_std == 0:
                return 0.0
            return abs(value - self.timing_mean) / self.timing_std
        else:
            return 0.0


class BaselineManager:
    """
    Manages baselines for multiple agents.

    Computes baselines from activity history, updates them incrementally,
    and provides baseline lookups for detection.

    Example:
        manager = BaselineManager()
        manager.record_activity(activity)
        baseline = manager.get_baseline("agent_001")
        z_score = baseline.z_score("entropy", observed_entropy)
    """

    def __init__(
        self,
        window_size: timedelta = timedelta(hours=24),
        min_samples: int = 10
    ):
        """
        Initialize baseline manager.

        Args:
            window_size: Time window for baseline calculation
            min_samples: Minimum samples needed for valid baseline
        """
        self._window_size = window_size
        self._min_samples = min_samples
        self._baselines: Dict[str, Baseline] = {}
        self._activity_history: Dict[str, List[Dict]] = defaultdict(list)

    def record_activity(self, activity: 'AgentActivity'):
        """
        Record an activity for baseline computation.

        Args:
            activity: The agent activity to record
        """
        record = {
            "timestamp": activity.timestamp,
            "activity_type": activity.activity_type,
            "entropy": activity.content.get("entropy"),
            "tool_calls": activity.content.get("tool_calls", []),
            "text": activity.content.get("text", ""),
        }

        self._activity_history[activity.agent_id].append(record)
        self._prune_old_records(activity.agent_id)

    def _prune_old_records(self, agent_id: str):
        """Remove records outside the window."""
        cutoff = datetime.now(timezone.utc) - self._window_size
        self._activity_history[agent_id] = [
            r for r in self._activity_history[agent_id]
            if r["timestamp"] > cutoff
        ]

    def get_baseline(self, agent_id: str) -> Baseline:
        """
        Get or compute baseline for an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Baseline for the agent
        """
        existing = self._baselines.get(agent_id)

        if existing and not existing.is_stale():
            return existing

        baseline = self._compute_baseline(agent_id)
        self._baselines[agent_id] = baseline
        return baseline

    def _compute_baseline(self, agent_id: str) -> Baseline:
        """Compute baseline from activity history."""
        history = self._activity_history.get(agent_id, [])

        if len(history) < self._min_samples:
            return Baseline(
                agent_id=agent_id,
                sample_count=len(history)
            )

        # Compute entropy statistics
        entropies = [
            r["entropy"] for r in history
            if r.get("entropy") is not None
        ]
        entropy_mean = statistics.mean(entropies) if entropies else 2.5
        entropy_std = (
            statistics.stdev(entropies) if len(entropies) > 1 else 0.5
        )

        # Compute timing statistics
        timings = []
        sorted_history = sorted(history, key=lambda x: x["timestamp"])
        for i in range(1, len(sorted_history)):
            delta = (
                sorted_history[i]["timestamp"] -
                sorted_history[i-1]["timestamp"]
            ).total_seconds() * 1000
            timings.append(delta)

        timing_mean = statistics.mean(timings) if timings else 1000.0
        timing_std = statistics.stdev(timings) if len(timings) > 1 else 500.0

        # Compute tool call patterns
        tool_patterns = self._compute_tool_patterns(history)

        # Compute structure patterns
        structure_patterns = self._compute_structure_patterns(history)

        return Baseline(
            agent_id=agent_id,
            entropy_mean=entropy_mean,
            entropy_std=entropy_std,
            tool_call_patterns=tool_patterns,
            structure_patterns=structure_patterns,
            timing_mean=timing_mean,
            timing_std=timing_std,
            sample_count=len(history),
            last_updated=datetime.now(timezone.utc)
        )

    def _compute_tool_patterns(
        self,
        history: List[Dict]
    ) -> Dict[str, float]:
        """Compute tool call pattern frequencies."""
        patterns: Dict[str, int] = defaultdict(int)
        total = 0

        tool_records = []
        for record in history:
            tool_calls = record.get("tool_calls", [])
            for call in tool_calls:
                if isinstance(call, dict):
                    name = call.get("tool_name") or call.get("name")
                    if name:
                        tool_records.append(name)

        for i in range(len(tool_records) - 1):
            pattern = f"{tool_records[i]}->{tool_records[i+1]}"
            patterns[pattern] += 1
            total += 1

        if total > 0:
            return {k: v / total for k, v in patterns.items()}
        return {}

    def _compute_structure_patterns(
        self,
        history: List[Dict]
    ) -> Dict[str, float]:
        """Compute structure pattern frequencies."""
        patterns: Dict[str, int] = defaultdict(int)
        total = 0

        for record in history:
            text = record.get("text", "")
            if not text:
                continue

            if text.strip().startswith(("1.", "- ", "* ")):
                pattern = "enumerated"
            elif "?" in text:
                pattern = "questioning"
            else:
                pattern = "prose"

            patterns[pattern] += 1
            total += 1

        if total > 0:
            return {k: v / total for k, v in patterns.items()}
        return {}

    def invalidate_baseline(self, agent_id: str):
        """Invalidate cached baseline for an agent."""
        if agent_id in self._baselines:
            del self._baselines[agent_id]

    def clear_history(self, agent_id: str):
        """Clear activity history for an agent."""
        if agent_id in self._activity_history:
            del self._activity_history[agent_id]
        self.invalidate_baseline(agent_id)

    def get_all_baselines(self) -> Dict[str, Baseline]:
        """Get all current baselines."""
        return dict(self._baselines)

    def __len__(self) -> int:
        """Return number of agents with baselines."""
        return len(self._baselines)
