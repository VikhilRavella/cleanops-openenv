"""
models.py — Typed Pydantic models for CleanOps OpenEnv.

Defines: Action, Observation, State, and supporting types.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from __future__ import annotations
from typing import Any, Dict, List, Optional  # <--- Make sure Optional is here!
from enum import Enum
from pydantic import BaseModel, Field

# ─── Action types ────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    inspect_column       = "inspect_column"
    standardize_format   = "standardize_format"
    normalize_dates      = "normalize_dates"
    validate_values      = "validate_values"
    replace_invalid      = "replace_invalid"
    fill_missing         = "fill_missing"
    remove_duplicates    = "remove_duplicates"
    merge_duplicates     = "merge_duplicates"
    apply_business_rule  = "apply_business_rule"
    finalize_cleaning    = "finalize_cleaning"


class Action(BaseModel):
    """Structured cleaning action sent by the agent to the environment."""

    action_type: ActionType = Field(
        ...,
        description="The type of cleaning action to perform.",
    )
    target_column: Optional[str] = Field(
        None,
        description="Column name to apply the action to. Not required for all action types.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Examples:\n"
            "  standardize_format: {format: 'title_case' | 'lower' | 'upper' | 'strip'}\n"
            "  normalize_dates:    {output_format: '%Y-%m-%d'}\n"
            "  replace_invalid:    {replacement: 'UNKNOWN'}\n"
            "  fill_missing:       {strategy: 'value', value: 'N/A'} | {strategy: 'mode'}\n"
            "  remove_duplicates:  {subset: ['col1', 'col2'], keep: 'first'}\n"
            "  apply_business_rule:{rule_name: 'fix_negative_quantities'}\n"
        ),
    )

    class Config:
        use_enum_values = True


# ─── Observation ─────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]


class QualityReport(BaseModel):
    total_rows: int
    duplicate_rows: int
    missing_value_count: int
    invalid_email_count: int
    invalid_phone_count: int
    invalid_date_count: int
    invalid_value_count: int
    quality_score: float = Field(..., ge=0.0, le=1.0)


class Observation(BaseModel):
    """What the agent sees after each step (or on reset)."""

    task_id: str
    task_name: str
    task_description: str
    difficulty: str
    table_preview: List[Dict[str, Any]]     # first 5 rows as list of dicts
    schema: List[ColumnInfo]
    quality_report: QualityReport
    allowed_actions: List[str]
    step_count: int
    max_steps: int
    message: str
    reward: float = 0.0
    done: bool = False


# ─── State ───────────────────────────────────────────────────────────────────

class State(BaseModel):
    """Full internal state of the environment episode."""

    task_id: str
    difficulty: str
    original_df: List[Dict[str, Any]]       # row-oriented JSON, immutable reference
    current_df: List[Dict[str, Any]]        # mutable current cleaned state
    expected_df: List[Dict[str, Any]]       # hidden target — used only by grader
    issue_summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step_count: int
    max_steps: int
    is_done: bool
    current_score_estimate: float = 0.0


# ─── API request / response models ───────────────────────────────────────────
class ResetRequest(BaseModel):
    # Adding Optional and None makes the body NOT required
    task_id: Optional[str] = Field(None, description="Task ID to start")

    class Config:
        extra = "allow"

class StepRequest(BaseModel):
    action: Action


class GraderRequest(BaseModel):
    task_id: str
    cleaned_rows: List[Dict[str, Any]] = Field(
        ..., description="The cleaned DataFrame as a list of row dicts."
    )


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    message: str


class TaskInfo(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    description: str
    max_steps: int
    expected_issue_types: List[str]


class BaselineResult(BaseModel):
    task_id: str
    difficulty: str
    score: float
    steps_taken: int
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]


class BaselineResponse(BaseModel):
    results: List[BaselineResult]
    overall_average: float
    summary: str
