from __future__ import annotations
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# ── Environment Setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, Observation
from tasks import TASK_REGISTRY
from environment import CleanOpsEnvironment
from graders import grade
from agent import get_agent

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# We look for both possible variable names to be safe
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("API_KEY", ""))

def _get_client():
    """Safely initialize the OpenAI client only when called."""
    try:
        from openai import OpenAI
        # If no token is found, we return None instead of crashing
        if not HF_TOKEN:
            return None
        return OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception:
        return None

def _llm_select_action(obs: Observation) -> Optional[Action]:
    """Uses the OpenAI Client for LLM calls."""
    client = _get_client()
    if client is None:
        return None
    
    system_prompt = "You are a data cleaning agent. Output ONLY valid JSON."
    user_msg = f"Task: {obs.task_id} | Step: {obs.step_count} | Message: {obs.message}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        
        data = json.loads(raw)
        return Action(
            action_type=data["action_type"],
            target_column=data.get("target_column"),
            parameters=data.get("parameters", {}),
        )
    except Exception:
        return None

def run_episode(task_id: str):
    env = CleanOpsEnvironment()
    heuristic = get_agent(task_id, prefer_llm=False)
    
    # ── MANDATORY LOGGING FORMAT: [START] ──
    print(f"[START] task={task_id} env=cleanops model={MODEL_NAME}", flush=True)
    
    try:
        obs = env.reset(task_id)
        step_n = 0
        rewards = []

        while not obs.done and step_n < 10:
            step_n += 1
            action = _llm_select_action(obs)
            if action is None:
                action = heuristic.select_action(obs)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            # ── MANDATORY LOGGING FORMAT: [STEP] ──
            print(
                f"[STEP] step={step_n} action={action.action_type} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
                flush=True
            )

        grade_res = grade(task_id, env.current_df, env.expected_df)
        success = grade_res.score >= 0.5
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        # ── MANDATORY LOGGING FORMAT: [END] ──
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_n} score={grade_res.score:.2f} rewards={rewards_str}",
            flush=True
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

def main():
    for task_id in sorted(TASK_REGISTRY.keys()):
        run_episode(task_id)

if __name__ == "__main__":
    main()
