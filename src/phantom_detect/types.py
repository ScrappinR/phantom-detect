"""
Core types for phantom-detect.

Defines channel types, detection alerts, and severity levels used
across the detection system.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime, timezone
import uuid


class ChannelType(Enum):
    """
    Types of covert channels detectable in LLM outputs.

    Categories:
        - Text-based: Encoding in word choice, structure, formatting
        - Timing-based: Encoding in response timing patterns
        - Tool-based: Encoding in tool call sequences
        - Multi-agent: Encoding via coordinated agent behavior
    """

    # Tool/API-level channels
    TCOE = auto()   # Tool Call Orchestration Encoding
    COTSE = auto()  # Chain-of-Thought Structure Encoding
    HSCE = auto()   # Hardware Side-Channel Encoder (timing)

    # Statistical/distribution channels
    EGE = auto()    # Entropy Gradient Encoding
    PRS = auto()    # Probability Ratio Steganography

    # Deep model channels
    AGE = auto()    # Attention Geometry Engine
    LSPE = auto()   # Latent Space Path Encoding

    # Multi-agent channels
    CASP = auto()   # Cross-Agent Synchronization Protocol

    # Text steganography channels
    PUNC = auto()   # Punctuation Pattern Encoding
    CITE = auto()   # Citation Order Steganography
    SYNC = auto()   # Response Timing Synchronization
    CASE = auto()   # Capitalization Micro-Patterns
    WHITE = auto()  # Whitespace Encoding


class DetectionSeverity(Enum):
    """Severity levels for detected covert channels."""

    INFO = auto()      # Informational, likely false positive
    LOW = auto()       # Possible covert channel activity
    MEDIUM = auto()    # Probable covert channel activity
    HIGH = auto()      # Confirmed covert channel activity
    CRITICAL = auto()  # Active hostile communication detected


@dataclass
class DetectionAlert:
    """
    Alert generated when covert channel activity is detected.

    Attributes:
        alert_id: Unique identifier for this alert
        severity: How serious is this detection
        channel_type: What type of channel was detected (if known)
        description: Human-readable description
        evidence: Supporting data for the detection
        timestamp: When detection occurred
        source_agent_id: Which agent triggered this alert
        recommended_action: What should be done
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: DetectionSeverity = DetectionSeverity.INFO
    channel_type: Optional[ChannelType] = None
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent_id: Optional[str] = None
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.name,
            "channel_type": self.channel_type.name if self.channel_type else None,
            "description": self.description,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "source_agent_id": self.source_agent_id,
            "recommended_action": self.recommended_action,
        }
