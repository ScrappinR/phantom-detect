"""
phantom-detect: LLM Covert Channel Detection Toolkit

Detects steganographic covert channels in AI/LLM outputs through
statistical anomaly detection, behavioral baselining, and cross-agent
correlation analysis.

Quick start:
    from phantom_detect import CCDS, AgentActivity
    from datetime import datetime, timezone

    ccds = CCDS(sensitivity=0.7)
    activity = AgentActivity(
        agent_id="agent_001",
        timestamp=datetime.now(timezone.utc),
        activity_type="generation",
        content={"text": "Hello world", "entropy": 3.2}
    )
    alerts = ccds.analyze_activity(activity)
"""

__version__ = "0.1.0"

from .ccds import CCDS, AgentActivity
from .baseline import Baseline, BaselineManager
from .statistical import StatisticalDetector, AnomalyScore
from .types import ChannelType, DetectionAlert, DetectionSeverity

__all__ = [
    "CCDS",
    "AgentActivity",
    "Baseline",
    "BaselineManager",
    "StatisticalDetector",
    "AnomalyScore",
    "ChannelType",
    "DetectionAlert",
    "DetectionSeverity",
]
