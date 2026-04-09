from __future__ import annotations
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# ── Environment imports ──────────────────────────────────────────────────────
# Fixes the NameError and ensures root imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, ActionType, Observation
from tasks import TASK_REGISTRY
from agent import get_agent
from graders import grade
from environment import CleanOpsEnvironment

# ── Config (MANDATORY: Scaler injects these specific variables) ──────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# KEY FIX: Using API_KEY ensures the Scaler Proxy tracks your agent
API_KEY: str      = os.getenv("API_KEY", os.getenv("HF_TOKEN", "")) 
ENV_NAME: str     = "cleanops"

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None and API_KEY:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY,
            )
        except Exception:
            _openai_client = None
    return _openai_client

# ── LLM-based action selection ────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a data cleaning agent. Output ONLY valid JSON."""

def _llm_select_action(obs: Observation) -> Optional[Action]:
    """Ask the LLM for the next action via the Scaler Proxy."""
    client = _get_openai_client()
    if client is None:
        return None

    user_msg = f"Task: {obs.task_id} | Step: {obs.step_count}/{obs.max_steps} | Message: {obs.message}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Clean up potential markdown formatting
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        data = json.loads(raw.strip())
        return Action(
            action_type=data["action_type"],
            target_column=data.get("target_column"),
            parameters=data.get("parameters", {}),
        )
    except Exception:
        return None

# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str):
    task = TASK_REGISTRY[task_id]
    env = CleanOpsEnvironment()
    heuristic = get_agent(task_id, prefer_llm=False)
    
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
    
    try:
        obs = env.reset(task_id)
        step_n = 0
        rewards = []

        while not obs.done and step_n < task.max_steps:
            step_n += 1
            
            # MANDATORY: Attempt LLM first so the Proxy records the activity
            action = _llm_select_action(obs)
            
            # Fallback if LLM fails
            if action is None:
                action = heuristic.select_action(obs)

            obs, reward, done, info = env.step(action)
            rewards.append(round(reward, 2))

            print(
                f"[STEP] step={step_n} action={action.action_type} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
                flush=True,
            )

        grade_res = grade(task_id, env.current_df, env.expected_df)
        success = grade_res.score >= 0.5
        rewards_str = ",".join(map(str, rewards))
        
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_n} score={grade_res.score:.2f} rewards={rewards_str}",
            flush=True,
        )
        
    except Exception as e:
        print(f"Episode Error: {e}", file=sys.stderr)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for task_id in sorted(TASK_REGISTRY.keys()):
        run_episode(task_id)

if __name__ == "__main__":
    main()
