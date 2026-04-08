from __future__ import annotations
import os
import sys
from typing import List, Optional

# Add current directory to path so it can find environment.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# IMPORTANT: Ensure your models.py, tasks.py, and graders.py are in the SAME folder
from models import (
    Observation, State, ResetRequest, StepRequest, 
    GraderRequest, GraderResponse, TaskInfo, BaselineResponse
)
from tasks import list_tasks, get_task
from graders import grade
from environment import CleanOpsEnvironment

app = FastAPI(title="CleanOps OpenEnv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = CleanOpsEnvironment()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/tasks", response_model=List[TaskInfo])
def get_tasks_list():
    return list_tasks()

@app.post("/reset", response_model=Observation)
def reset_episode(request: Optional[ResetRequest] = None):
    try:
        # Defaulting to task_001 as required by the grader
        target_task = "task_001"
        if request and request.task_id:
            target_task = request.task_id
        return _env.reset(target_task)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=Observation)
def step_action(request: StepRequest):
    try:
        obs, reward, done, info = _env.step(request.action)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=State)
def get_state():
    return _env.state()

@app.post("/grader", response_model=GraderResponse)
def grade_submission(request: GraderRequest):
    try:
        task = get_task(request.task_id)
        cleaned_df = pd.DataFrame(request.cleaned_rows)
        return grade(request.task_id, cleaned_df, task.get_expected_df())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use "app:app" because the file is app.py
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
