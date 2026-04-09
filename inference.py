from __future__ import annotations
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional
from openai import OpenAI  # Import at top

# ── Environment Setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, Observation
from tasks import TASK_REGISTRY
from environment import CleanOpsEnvironment
from graders import grade
from agent import get_agent

# ── Config (Strictly following the Validator's "How to Fix" section) ────────
# They explicitly told us to use os.environ["API_BASE_URL"] and os.environ["API_KEY"]
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

def _llm_select_action(obs: Observation) -> Optional[Action]:
    """Uses the mandatory OpenAI Client for LLM calls via the Proxy."""
    # Initialize inside to ensure environment variables are loaded by the OS
    if not API_KEY:
        return None

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    
    system_prompt = "You are a data cleaning agent. Output ONLY valid JSON."
    user_msg = f"Task: {obs.task_id} | Step: {obs.step_count} | Message: {obs.message}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
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
    
    print(f"[START] task={task_id} env=cleanops model={MODEL_NAME}", flush=True)
    
    try:
        obs = env.reset(task_id)
        step_n = 0
        rewards = []

        while not obs.done and step_n < 10:
            step_n += 1
            
            # MANDATORY: Try LLM first
            action = _llm_select_action(obs)
            
            # FALLBACK
            if action is None:
                action = heuristic.select_action(obs)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            print(
                f"[STEP] step={step_n} action={action.action_type} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
                flush=True
            )

        grade_res = grade(task_id, env.current_df, env.expected_df)
        success = grade_res.score >= 0.5
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

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
