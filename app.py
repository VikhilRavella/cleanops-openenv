"""
server/app.py — FastAPI application for CleanOps OpenEnv.

Endpoints:
    GET  /health
    GET  /tasks
    POST /reset
    POST /step
    GET  /state
    POST /grader
    POST /baseline
"""

from __future__ import annotations  # <--- MUST BE LINE 1
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
# ... the rest of your imports ...
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from __future__ import annotations
import os
from typing import Dict, List, Optional  # <--- Add Optional here too!
from fastapi import FastAPI, HTTPException
from typing import Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Action,
    Observation,
    State,
    ResetRequest,
    StepRequest,
    GraderRequest,
    GraderResponse,
    TaskInfo,
    BaselineResponse,
)
from tasks import list_tasks, get_task
from graders import grade
from server.environment import CleanOpsEnvironment


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CleanOps OpenEnv",
    description=(
        "A multi-task AI agent environment for solving real-world data cleaning, "
        "normalization, validation, and quality repair problems across messy business datasets."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (one episode at a time)
_env = CleanOpsEnvironment()


# ─── /health ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Returns server health status."""
    return {"status": "healthy", "service": "CleanOps OpenEnv", "version": "1.0.0"}


# ─── /tasks ──────────────────────────────────────────────────────────────────

@app.get("/tasks", response_model=List[TaskInfo], tags=["Tasks"])
def get_tasks():
    """List all available benchmark tasks with metadata and action schema."""
    task_list = list_tasks()
    action_schema = {
        "action_type": "One of: inspect_column | standardize_format | normalize_dates | "
                       "validate_values | replace_invalid | fill_missing | remove_duplicates | "
                       "merge_duplicates | apply_business_rule | finalize_cleaning",
        "target_column": "Column name (string, optional for some actions)",
        "parameters": {
            "standardize_format":  {"format": "strip | lower | upper | title_case"},
            "normalize_dates":     {"output_format": "%Y-%m-%d"},
            "validate_values":     {"validator": "email | phone | date"},
            "replace_invalid":     {"validator": "email | phone", "replacement": "null or string"},
            "fill_missing":        {"strategy": "value | mode | ffill | bfill", "value": "fill value"},
            "remove_duplicates":   {"subset": ["col1", "col2"], "keep": "first | last"},
            "merge_duplicates":    {"key_column": "column name"},
            "apply_business_rule": {
                "rule_name": (
                    "fix_negative_quantities | fix_negative_prices | normalize_currency | "
                    "fix_missing_employee_ids | fix_total_comp | fix_state_country"
                )
            },
        },
    }
    return [
        TaskInfo(
            task_id=t["task_id"],
            task_name=t["task_name"],
            difficulty=t["difficulty"],
            description=t["description"],
            max_steps=t["max_steps"],
            expected_issue_types=t["expected_issue_types"],
        )
        for t in task_list
    ]


# ─── /reset ──────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, tags=["Environment"])
def reset_episode(request: Optional[ResetRequest] = None):
    try:
        # If the whole request is None or task_id is None, use "task_1"
        target_task = "task_1"
        if request and request.task_id:
            target_task = request.task_id
            
        obs = _env.reset(target_task)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# ─── /step ───────────────────────────────────────────────────────────────────

@app.post("/step", response_model=Observation, tags=["Environment"])
def step_action(request: StepRequest):
    """
    Submit a structured cleaning action. Returns updated observation + reward.
    """
    try:
        obs, reward, done, info = _env.step(request.action)
        return obs
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── /state ──────────────────────────────────────────────────────────────────

@app.get("/state", response_model=State, tags=["Environment"])
def get_state():
    """
    Return the full current environment state including
    original_df, current_df, expected_df, action history, and score estimate.
    """
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── /grader ─────────────────────────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse, tags=["Grader"])
def grade_submission(request: GraderRequest):
    """
    Grade a cleaned dataset against the hidden expected output.
    Returns score [0.0, 1.0] with detailed breakdown.

    This is the FINAL grader — separate from step-level reward.
    """
    try:
        task = get_task(request.task_id)
        cleaned_df  = pd.DataFrame(request.cleaned_rows)
        expected_df = task.get_expected_df()
        return grade(request.task_id, cleaned_df, expected_df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading error: {e}")


# ─── /baseline ───────────────────────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
def run_baseline():
    """
    Run the heuristic baseline agent on all 3 tasks.
    Returns per-task scores, metrics, and overall average.
    No API key required — uses deterministic heuristic agent.
    """
    from baseline import run_all_tasks
    results = run_all_tasks()
    return results


# ─── Run (dev) ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
