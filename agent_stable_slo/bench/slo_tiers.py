"""SLO tier definitions for AgentSLO-Bench.

Three deployment tiers with different latency budgets:
- Interactive: 2s (chatbots, autocomplete)
- Standard: 5s (API backends, async pipelines)
- Batch: 30s (offline processing, nightly jobs)
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SLOTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    deadline_ms: float
    description: str
    typical_use: str


INTERACTIVE = SLOTier(
    name="interactive",
    deadline_ms=2000.0,
    description="Real-time user-facing applications",
    typical_use="Chatbots, autocomplete, inline suggestions",
)

STANDARD = SLOTier(
    name="standard",
    deadline_ms=5000.0,
    description="API-driven backend services",
    typical_use="REST APIs, async pipelines, webhook handlers",
)

BATCH = SLOTier(
    name="batch",
    deadline_ms=30000.0,
    description="Offline and batch processing",
    typical_use="Nightly jobs, bulk classification, data enrichment",
)

TIERS = [INTERACTIVE, STANDARD, BATCH]
TIER_MAP = {t.name: t for t in TIERS}
