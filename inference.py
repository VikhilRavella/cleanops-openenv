from __future__ import annotations
import json
import os
import sys
import traceback
from typing import List, Optional
from openai import OpenAI

# ── Environment Setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything except the agent first
from models import Action, Observation
from tasks import TASK_REGISTRY
from environment import CleanOpsEnvironment
from graders import grade

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Global client variable
client = None

def _get_client():
    global client
    if client is None and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client

def _llm_select_action(obs: Observation) -> Optional[Action]:
    c = _get_client()
    if not c: return None
    
    system_prompt = "You are a data cleaning agent. Output ONLY valid JSON."
    user_msg = f"Task: {obs.task_id} | Message: {obs.message}"

    try:
        response = c.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
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

def run_episode(task_id: str, agent_factory):
    env = CleanOpsEnvironment()
    heuristic = agent_factory(task_id, prefer_llm=False)

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

            print(f"[STEP] step={step_n} action={action.action_type} reward={reward:.2f} done={done} error=null", flush=True)

        grade_res = grade(task_id, env.current_df, env.expected_df)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={grade_res.score >= 0.5} steps={step_n} score={grade_res.score:.2f} rewards={rewards_str}", flush=True)

    except Exception as e:
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        print(f"# Error: {e}", file=sys.stderr)

def main():
    # Import agent inside main to break circular dependency
    from agent import get_agent
    
    for task_id in sorted(TASK_REGISTRY.keys()):
        run_episode(task_id, get_agent)

if __name__ == "__main__":
    main()
