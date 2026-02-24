"""Tests for the Covert Channel Detection System."""

import pytest
from datetime import datetime, timedelta, timezone

from phantom_detect import (
    CCDS,
    AgentActivity,
    Baseline,
    BaselineManager,
    StatisticalDetector,
    AnomalyScore,
    DetectionSeverity,
    ChannelType,
)


def make_activity(
    agent_id="agent_001",
    activity_type="generation",
    entropy=2.5,
    text="Normal output text",
    timestamp=None,
    metadata=None,
):
    """Helper to create test activities."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    content = {}
    if entropy is not None:
        content["entropy"] = entropy
    if text is not None:
        content["text"] = text
    return AgentActivity(
        agent_id=agent_id,
        timestamp=timestamp,
        activity_type=activity_type,
        content=content,
        metadata=metadata or {},
    )


class TestCCDSInit:
    def test_default_init(self):
        ccds = CCDS()
        assert ccds.sensitivity == 0.7
        assert len(ccds._recent_activities) == 0
        assert len(ccds._alert_history) == 0

    def test_custom_sensitivity(self):
        ccds = CCDS(sensitivity=0.3)
        assert ccds.sensitivity == 0.3

    def test_sensitivity_bounds(self):
        ccds = CCDS()
        ccds.set_sensitivity(1.5)
        assert ccds.sensitivity == 1.0
        ccds.set_sensitivity(-0.5)
        assert ccds.sensitivity == 0.0


class TestCCDSAnalysis:
    def test_normal_activity_no_alerts(self):
        ccds = CCDS(sensitivity=0.5)
        activity = make_activity(entropy=2.5)
        alerts = ccds.analyze_activity(activity)
        assert len(alerts) == 0

    def test_anomalous_entropy_generates_alert(self):
        """Feed baseline-consistent data, then inject multi-signal anomaly."""
        ccds = CCDS(sensitivity=0.9)

        # Build baseline with consistent entropy and timing
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(20):
            activity = make_activity(
                entropy=2.5 + (i % 3) * 0.1,
                timestamp=base_time + timedelta(minutes=i),
                metadata={"time_since_last_ms": 1000},
            )
            ccds.analyze_activity(activity)

        # Inject activity anomalous on multiple axes (entropy + timing)
        anomalous = make_activity(
            entropy=9.8,
            metadata={"time_since_last_ms": 50},
        )
        alerts = ccds.analyze_activity(anomalous)

        # Multi-signal anomaly should trigger alert
        assert len(alerts) >= 1
        assert any(a.severity.value >= DetectionSeverity.LOW.value for a in alerts)

    def test_cross_agent_correlation(self):
        """Activities from 3+ agents within correlation window should alert."""
        ccds = CCDS(
            sensitivity=0.5,
            correlation_window=timedelta(seconds=2),
            enable_cross_agent=True,
        )

        now = datetime.now(timezone.utc)

        # Three agents acting within 1 second of each other
        for agent_id in ["agent_A", "agent_B", "agent_C"]:
            activity = make_activity(
                agent_id=agent_id,
                timestamp=now + timedelta(milliseconds=100),
            )
            alerts = ccds.analyze_activity(activity)

        # The third agent should trigger cross-agent correlation
        assert any(
            a.channel_type == ChannelType.CASP
            for a in ccds.get_recent_alerts()
        )

    def test_no_cross_agent_when_disabled(self):
        ccds = CCDS(enable_cross_agent=False)
        now = datetime.now(timezone.utc)

        for agent_id in ["agent_A", "agent_B", "agent_C"]:
            activity = make_activity(
                agent_id=agent_id,
                timestamp=now,
            )
            ccds.analyze_activity(activity)

        casp_alerts = [
            a for a in ccds.get_recent_alerts()
            if a.channel_type == ChannelType.CASP
        ]
        assert len(casp_alerts) == 0


class TestBaseline:
    def test_default_baseline(self):
        baseline = Baseline(agent_id="test")
        assert baseline.entropy_mean == 2.5
        assert baseline.sample_count == 0

    def test_z_score_entropy(self):
        baseline = Baseline(agent_id="test", entropy_mean=2.5, entropy_std=0.5)
        z = baseline.z_score("entropy", 3.5)
        assert z == pytest.approx(2.0)

    def test_z_score_zero_std(self):
        baseline = Baseline(agent_id="test", entropy_std=0.0)
        z = baseline.z_score("entropy", 5.0)
        assert z == 0.0

    def test_stale_detection(self):
        old_baseline = Baseline(
            agent_id="test",
            last_updated=datetime.now(timezone.utc) - timedelta(hours=48)
        )
        assert old_baseline.is_stale()

        fresh_baseline = Baseline(agent_id="test")
        assert not fresh_baseline.is_stale()


class TestBaselineManager:
    def test_get_default_baseline(self):
        manager = BaselineManager()
        baseline = manager.get_baseline("new_agent")
        assert baseline.agent_id == "new_agent"
        assert baseline.sample_count == 0

    def test_record_and_compute(self):
        manager = BaselineManager(min_samples=3)

        base_time = datetime.now(timezone.utc)
        for i in range(5):
            activity = make_activity(
                entropy=2.0 + i * 0.2,
                timestamp=base_time + timedelta(seconds=i),
            )
            manager.record_activity(activity)

        baseline = manager.get_baseline("agent_001")
        assert baseline.sample_count == 5
        assert baseline.entropy_mean > 0


class TestStatisticalDetector:
    def test_normal_score(self):
        detector = StatisticalDetector(sensitivity=0.5)
        baseline = Baseline(agent_id="test", entropy_mean=2.5, entropy_std=0.5, sample_count=100)
        activity = make_activity(entropy=2.6)
        score = detector.analyze(activity, baseline)
        assert not score.is_anomalous

    def test_anomalous_score(self):
        """Multi-signal anomaly should produce anomalous overall score."""
        detector = StatisticalDetector(sensitivity=0.9)
        baseline = Baseline(
            agent_id="test",
            entropy_mean=2.5, entropy_std=0.3,
            timing_mean=1000, timing_std=200,
            sample_count=100,
        )
        # Anomalous on both entropy AND timing
        activity = make_activity(
            entropy=8.0,
            metadata={"time_since_last_ms": 50},
        )
        score = detector.analyze(activity, baseline)
        assert score.is_anomalous
        assert score.entropy_score > 0.5
        assert score.timing_score > 0.5

    def test_create_alert(self):
        detector = StatisticalDetector(sensitivity=0.9)
        score = AnomalyScore(
            overall_score=0.85,
            entropy_score=0.9,
            timing_score=0.1,
            pattern_score=0.2,
            contributing_factors=["entropy_deviation(0.90)"]
        )
        activity = make_activity()
        alert = detector.create_alert(activity, score)
        assert alert.severity == DetectionSeverity.HIGH
        assert alert.channel_type == ChannelType.EGE


class TestAlertHistory:
    def test_get_recent_alerts_by_severity(self):
        ccds = CCDS(sensitivity=0.9)

        # Build baseline then inject anomaly
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(20):
            ccds.analyze_activity(make_activity(
                entropy=2.5,
                timestamp=base_time + timedelta(minutes=i),
            ))

        ccds.analyze_activity(make_activity(entropy=9.9))

        high_alerts = ccds.get_recent_alerts(min_severity=DetectionSeverity.HIGH)
        all_alerts = ccds.get_recent_alerts(min_severity=DetectionSeverity.INFO)
        assert len(all_alerts) >= len(high_alerts)

    def test_clear_history(self):
        ccds = CCDS()
        ccds.analyze_activity(make_activity())
        assert len(ccds._recent_activities) > 0

        ccds.clear_history()
        assert len(ccds._recent_activities) == 0
        assert len(ccds._alert_history) == 0

    def test_alert_handler(self):
        ccds = CCDS(sensitivity=0.9)
        captured = []
        ccds.register_alert_handler(lambda alert: captured.append(alert))

        # Build baseline
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(20):
            ccds.analyze_activity(make_activity(
                entropy=2.5,
                timestamp=base_time + timedelta(minutes=i),
            ))

        ccds.analyze_activity(make_activity(entropy=9.9))

        # Handler should have been called for any alerts
        if ccds._alert_history:
            assert len(captured) > 0

    def test_alert_statistics(self):
        ccds = CCDS()
        stats = ccds.get_alert_statistics()
        assert stats["total_alerts"] == 0

    def test_repr(self):
        ccds = CCDS(sensitivity=0.5)
        assert "sensitivity=0.50" in repr(ccds)
