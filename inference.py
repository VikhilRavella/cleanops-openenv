"""
inference.py — CleanOps OpenEnv Inference Script
===================================================

MANDATORY SUBMISSION REQUIREMENTS:
  - API_BASE_URL   : The API endpoint for the LLM (also used as env base URL)
  - MODEL_NAME     : The model identifier for LLM inference
  - HF_TOKEN       : Your Hugging Face / API key
  - Uses OpenAI Client for all LLM calls
  - Emits exactly [START], [STEP], [END] lines to stdout

STDOUT FORMAT:
  [START] task=<task_id> env=cleanops model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

USAGE:
  # With LLM agent (requires API key):
  API_BASE_URL=https://api-inference.huggingface.co/v1 \
  MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  HF_TOKEN=hf_xxx \
  python inference.py

  # Heuristic fallback (no API key needed):
  python inference.py
"""
from __future__ import annotations
import json
import os
import sys  # <--- MAKE SURE THIS IS HERE
import traceback
from typing import Any, Dict, List, Optional

# Now this line will work perfectly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, ActionType, Observation
from tasks import TASK_REGISTRY
from agent import get_agent
from graders import grade
from environment import CleanOpsEnvironment # <--- Also make sure 'server.' is removed here!
# ── Config (from env vars) ───────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")
ENV_NAME: str     = "cleanops"

# ── Optional OpenAI client (used only when HF_TOKEN is available) ─────────────
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None and HF_TOKEN:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN,
            )
        except Exception:
            _openai_client = None
    return _openai_client


# ── LLM-based action selection ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a data cleaning agent. Given an observation of a messy dataset,
you must output a single JSON action to clean it.

Available action_types:
  inspect_column, standardize_format, normalize_dates, validate_values,
  replace_invalid, fill_missing, remove_duplicates, merge_duplicates,
  apply_business_rule, finalize_cleaning

Return ONLY valid JSON with this exact shape (no extra text):
{
  "action_type": "<action_type>",
  "target_column": "<column_name_or_null>",
  "parameters": {}
}

Examples:
{"action_type": "standardize_format", "target_column": "name", "parameters": {"format": "title_case"}}
{"action_type": "remove_duplicates", "target_column": null, "parameters": {"subset": null, "keep": "first"}}
{"action_type": "normalize_dates", "target_column": "date", "parameters": {"output_format": "%Y-%m-%d"}}
{"action_type": "fill_missing", "target_column": "email", "parameters": {"strategy": "value", "value": null}}
{"action_type": "apply_business_rule", "target_column": null, "parameters": {"rule_name": "fix_negative_quantities"}}
{"action_type": "finalize_cleaning", "target_column": null, "parameters": {}}
"""


def _llm_select_action(obs: Observation, history: List[str]) -> Optional[Action]:
    """Ask the LLM for the next action. Returns None on failure."""
    client = _get_openai_client()
    if client is None:
        return None

    # Build concise observation summary for the prompt
    quality = obs.quality_report
    schema_cols = [f"{c.name}({c.dtype}, nulls={c.null_count})" for c in obs.schema]

    user_msg = f"""Task: {obs.task_id} | Step: {obs.step_count}/{obs.max_steps}
Columns: {', '.join(schema_cols)}
Quality: rows={quality.total_rows}, dups={quality.duplicate_rows}, missing={quality.missing_value_count}
         bad_emails={quality.invalid_email_count}, bad_phones={quality.invalid_phone_count}
         bad_dates={quality.invalid_date_count}, quality_score={quality.quality_score:.3f}
Message: {obs.message}
Done: {obs.done}
History (last 5): {history[-5:]}

Output the next single cleaning action as JSON:"""

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
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return Action(
            action_type=data["action_type"],
            target_column=data.get("target_column"),
            parameters=data.get("parameters", {}),
        )
    except Exception as exc:
        # LLM failed — fall back silently
        return None


# ── Single task episode runner ─────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Run one full episode for a task.
    Returns dict with: success, steps, score, rewards, error
    Emits [START], [STEP]..., [END] to stdout.
    """
    task       = TASK_REGISTRY[task_id]
    env        = CleanOpsEnvironment()
    heuristic  = get_agent(task_id, prefer_llm=False)

    rewards: List[float] = []
    last_error: Optional[str] = None
    step_n = 0
    score  = 0.0
    success = False

    # Print [START]
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id)
        heuristic.plan(obs)

        action_history: List[str] = []

        while not obs.done and step_n < task.max_steps:
            step_n += 1

            # Try LLM first, fall back to heuristic
            action: Optional[Action] = None
            if use_llm and HF_TOKEN:
                action = _llm_select_action(obs, action_history)

            if action is None:
                action = heuristic.select_action(obs)

            # Represent action as compact string for logging
            action_str = (
                f"{action.action_type}"
                f"({action.target_column or ''}"
                f"{(',' + str(action.parameters)) if action.parameters else ''})"
            )
            action_history.append(action_str)

            # Step the environment
            obs, reward, done, info = env.step(action)
            rewards.append(round(reward, 2))
            # Detect errors from negative reward or "failed" in message
            last_error = None
            if reward < 0 and obs.message and ("fail" in obs.message.lower() or "error" in obs.message.lower()):
                last_error = obs.message[:80].replace("\n", " ")

            # Print [STEP]
            print(
                f"[STEP] step={step_n} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={last_error if last_error else 'null'}",
                flush=True,
            )

        # Final grader score (separate from step rewards)
        grade_response = grade(task_id, env.current_df, env.expected_df)
        score   = round(grade_response.score, 2)
        success = score >= 0.5

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)

    # Always print [END]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_n} "
        f"score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps":   step_n,
        "score":   score,
        "rewards": rewards,
        "error":   last_error,
    }


# ── Main: run all 3 tasks ──────────────────────────────────────────────────────

def main():
    use_llm = bool(HF_TOKEN)

    if use_llm:
        print(f"# Mode: LLM agent  |  model={MODEL_NAME}  |  endpoint={API_BASE_URL}", flush=True)
    else:
        print("# Mode: Heuristic fallback  (no HF_TOKEN set — LLM disabled)", flush=True)

    all_results = []
    for task_id in sorted(TASK_REGISTRY.keys()):
        result = run_episode(task_id, use_llm=use_llm)
        all_results.append(result)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n# ── Summary ──────────────────────────────", flush=True)
    for r in all_results:
        print(f"#  {r['task_id']}  score={r['score']:.2f}  steps={r['steps']}", flush=True)
    print(f"#  Overall average: {avg_score:.2f}", flush=True)


if __name__ == "__main__":
    main()
