from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

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
    action_type: ActionType
    target_column: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        use_enum_values = True

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
    quality_score: float

class Observation(BaseModel):
    task_id: str
    task_name: str
    task_description: str
    difficulty: str
    table_preview: List[Dict[str, Any]]
    columns: List[ColumnInfo]          # ← changed from schema
    quality_report: QualityReport
    allowed_actions: List[str]
    step_count: int
    max_steps: int
    message: str
    reward: float = 0.0
    done: bool = False

class State(BaseModel):
    task_id: str
    difficulty: str
    original_df: List[Dict[str, Any]]
    current_df: List[Dict[str, Any]]
    expected_df: List[Dict[str, Any]]
    issue_summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step_count: int
    max_steps: int
    is_done: bool
    current_score_estimate: float = 0.0

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field("task_001", description="Task ID to start")
    class Config:
        extra = "allow"

class StepRequest(BaseModel):
    action: Action

class GraderRequest(BaseModel):
    task_id: str
    cleaned_rows: List[Dict[str, Any]]

class GraderResponse(BaseModel):
    task_id: str
    score: float
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
