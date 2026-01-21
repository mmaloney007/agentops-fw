from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    task_file: str
    output_schema: str


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: List[TaskSpec]


class GatingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: List[str]
    policy: str = "strict"


class SecondaryJudgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens_out: int = 512


class JudgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens_out: int = 512
    prompt_version: str = "faithfulness_atomic_v1"
    secondary_judge: Optional[SecondaryJudgeSpec] = None


class CalibrationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    expected_calibration_error_max: float = 0.10
    confidence_field: str = "confidence"


class AnswerRelevancySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    hotpot_answer_relevancy_min: float = 0.80


class ContextQualitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    context_precision_min: float = 0.60
    context_recall_min: float = 0.70
    evidence_field: str = "evidence"


class StructureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_valid_min: float = 0.99
    schema_valid_min: float = 0.98
    validation_retry_rate_max: float = 0.05
    output_repair_rate_max: float = 0.02


class AccuracySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinc_intent_macro_f1_min: float = 0.85
    clinc_intent_accuracy_min: float = 0.85
    hotpot_answer_exact_match_min: float = 0.65
    hotpot_answer_f1_min: float = 0.75
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)


class FaithfulnessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotpot_faithfulness_min: float = 0.80
    hotpot_contradiction_rate_max: float = 0.05
    answer_relevancy: AnswerRelevancySpec = Field(default_factory=AnswerRelevancySpec)
    context_quality: ContextQualitySpec = Field(default_factory=ContextQualitySpec)


class ToolsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_success_rate_min: float = 0.90
    tool_argument_schema_valid_rate_min: float = 0.98
    tool_calls_p95_max: int = 4


class StabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disagreement_max: float = 0.10
    runs_per_prompt: int = 5
    equivalence: str = "canonical_json"


class SloSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p95_ms_max: float = 1200.0
    p99_ms_max: float = 2000.0
    max_tokens_out: int = 512
    hard_timeout_ms: int = 5000
    success_at_slo_min: float = 0.75
    on_time_budget_ms: int = 1200


class TierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    extends: Optional[str] = None
    must_pass: List[str] = Field(default_factory=list)


class WandbReportingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "online"
    entity: Optional[str] = None
    project: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    log_artifacts: bool = True
    log_episode_table: bool = True
    artifact_alias: str = "latest"


class ReportingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb: WandbReportingSpec = Field(default_factory=WandbReportingSpec)


class BootstrapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    resamples: int = 1000
    confidence_level: float = 0.95


class ReproducibilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warmup_requests: int = 20
    min_episodes_for_p99_reporting: int = 1000
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)


class DocumentationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intended_papers: List[str] = Field(default_factory=list)
    typical_endpoints: List[str] = Field(default_factory=list)
    typical_models: List[str] = Field(default_factory=list)


class CriteriaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    criteria_id: str
    description: str = ""
    documentation: Optional[DocumentationSpec] = None
    suites: Dict[str, SuiteSpec]
    gating: GatingSpec
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    structure: StructureSpec = Field(default_factory=StructureSpec)
    accuracy: AccuracySpec = Field(default_factory=AccuracySpec)
    faithfulness: FaithfulnessSpec = Field(default_factory=FaithfulnessSpec)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    stability: StabilitySpec = Field(default_factory=StabilitySpec)
    slo: SloSpec = Field(default_factory=SloSpec)
    weights: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    tiers: Dict[str, TierSpec] = Field(default_factory=dict)
    reporting: ReportingSpec = Field(default_factory=ReportingSpec)
    reproducibility: ReproducibilitySpec = Field(default_factory=ReproducibilitySpec)
    notes: str = ""

    def criteria_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_criteria(path: str) -> CriteriaSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CriteriaSpec(**data)
