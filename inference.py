from __future__ import annotations
import json
import os
import sys
import traceback
from typing import List, Optional

# ── Environment Setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, Observation
from tasks import TASK_REGISTRY
from environment import CleanOpsEnvironment
from graders import grade

# Import get_agent safely
try:
    from agent import get_agent
except ImportError:
    def get_agent(*args, **kwargs):
        import agent
        return agent.get_agent(*args, **kwargs)

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── Global Client (Delayed Initialization) ──────────────────────────────────
_client = None

def _get_client():
    """Initializes the OpenAI client only when needed to prevent startup crashes."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            # We use a dummy key if API_KEY is missing to prevent the library from crashing
            _client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY if API_KEY else "no_key_found"
            )
        except Exception as e:
            print(f"# Client initialization failed: {e}", file=sys.stderr)
            return None
    return _client

def _llm_select_action(obs: Observation) -> Optional[Action]:
    """Call LLM via proxy to choose an action."""
    client = _get_client()
    if not client or not API_KEY:
        return None

    system_prompt = "You are a data cleaning agent. Output ONLY valid JSON."
    user_msg = f"Task: {obs.task_id} | Message: {obs.message}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
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
    except Exception as e:
        print(f"# LLM call failed: {e}", file=sys.stderr)
        return None

def run_episode(task_id: str):
    env = CleanOpsEnvironment()
    heuristic = get_agent(task_id, prefer_llm=False)

    print(f"[START] task={task_id} env=cleanops model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id)
        step_n = 0
        rewards: List[float] = []

        while not obs.done and step_n < 10:
            step_n += 1
            action = _llm_select_action(obs)
            if action is None:
                action = heuristic.select_action(obs)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            print(
                f"[STEP] step={step_n} action={action.action_type} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
                flush=True,
            )

        grade_res = grade(task_id, env.current_df, env.expected_df)
        success = grade_res.score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_n} score={grade_res.score:.2f} rewards={rewards_str}",
            flush=True,
        )

    except Exception as e:
        # Ensure we always print an [END] tag even if the loop crashes
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        print(f"# Error during episode: {e}", file=sys.stderr)

def main():
    for task_id in sorted(TASK_REGISTRY.keys()):
        run_episode(task_id)

if __name__ == "__main__":
    main()
