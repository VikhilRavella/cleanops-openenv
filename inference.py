# inference.py
from __future__ import annotations
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, Observation
from tasks import TASK_REGISTRY
from environment import CleanOpsEnvironment
from graders import grade
from agent import get_agent

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "dummy-key"
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _get_client():
    """Create OpenAI client lazily to avoid module-level crash."""
    from openai import OpenAI
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )


def _llm_select_action(obs: Observation) -> Optional[Action]:
    system_prompt = (
        "You are a data cleaning agent. Output ONLY valid JSON with keys: "
        "action_type (string), target_column (string or null), parameters (dict)."
    )
    user_msg = (
        f"Task: {obs.task_id}\n"
        f"Step: {obs.step_count}/{obs.max_steps}\n"
        f"Message: {obs.message}\n"
        f"Allowed act
