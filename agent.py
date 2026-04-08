"""
agent.py — Heuristic baseline agent for CleanOps OpenEnv.

Design principles:
  - Works with NO external API key (fully self-contained)
  - Deterministic rule-based action selection
  - Reads observations, selects structured actions
  - Does NOT mutate the DataFrame directly (that is environment.py's job)
  - Optional LLM-backed agent can be plugged in via OPENAI_API_KEY env var

Agent loop:
    obs = env.reset(task_id)
    while not obs.done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
    final_score = grader.grade(task_id, env.current_df, env.expected_df)
"""

from __future__ import annotations
import os
from typing import List, Optional

from models import Action, ActionType, Observation


# ─── Heuristic agent ─────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Deterministic rule-based agent.
    Reads the observation quality report and applies a fixed cleaning playbook.

    The playbook order:
      Phase 1 — Remove duplicates
      Phase 2 — Fill missing values
      Phase 3 — Standardize text columns
      Phase 4 — Normalize date columns
      Phase 5 — Fix phones and emails
      Phase 6 — Apply business rules
      Phase 7 — Finalize
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._phase_index = 0
        self._actions_queue: List[Action] = []

    def plan(self, obs: Observation) -> None:
        """Build a deterministic action queue based on the initial observation."""
        schema   = obs.schema
        qr       = obs.quality_report
        col_names = [c.name for c in schema]
        self._actions_queue = []

        def q(action: Action):
            self._actions_queue.append(action)

        # ── Phase 1: Remove duplicates ──
        q(Action(action_type=ActionType.remove_duplicates, parameters={"keep": "first"}))

        # ── Phase 2: Fill missing values ──
        for col_info in schema:
            if col_info.null_count > 0:
                col = col_info.name
                if any(k in col.lower() for k in ("id", "record", "invoice", "employee")):
                    q(Action(action_type=ActionType.apply_business_rule,
                             parameters={"rule_name": "fix_missing_employee_ids"}))
                elif any(k in col.lower() for k in ("category", "department", "type")):
                    q(Action(action_type=ActionType.fill_missing,
                             target_column=col,
                             parameters={"strategy": "value", "value": "UNKNOWN"}))
                elif any(k in col.lower() for k in ("product", "name")):
                    q(Action(action_type=ActionType.fill_missing,
                             target_column=col,
                             parameters={"strategy": "value", "value": "UNKNOWN"}))

        # ── Phase 3: Standardize text ──
        text_cols = [c.name for c in schema if c.dtype == "object"]
        for col in text_cols:
            col_lower = col.lower()
            if any(k in col_lower for k in ("name", "city", "department", "product", "category")):
                q(Action(action_type=ActionType.standardize_format,
                         target_column=col,
                         parameters={"format": "title_case"}))
            elif col_lower in ("email", "email_address"):
                q(Action(action_type=ActionType.standardize_format,
                         target_column=col,
                         parameters={"format": "lower"}))
            elif col_lower in ("currency",):
                pass  # handled by business rule
            else:
                q(Action(action_type=ActionType.standardize_format,
                         target_column=col,
                         parameters={"format": "strip"}))

        # ── Phase 4: Normalize dates ──
        date_cols = [c.name for c in schema if "date" in c.name.lower()]
        for col in date_cols:
            q(Action(action_type=ActionType.normalize_dates,
                     target_column=col,
                     parameters={"output_format": "%Y-%m-%d"}))

        # ── Phase 5: Fix phones ──
        phone_cols = [c.name for c in schema if "phone" in c.name.lower()]
        for col in phone_cols:
            q(Action(action_type=ActionType.validate_values,
                     target_column=col,
                     parameters={"validator": "phone"}))

        # Fix emails — replace invalid with None
        email_cols = [c.name for c in schema if "email" in c.name.lower()]
        for col in email_cols:
            if qr.invalid_email_count > 0:
                q(Action(action_type=ActionType.replace_invalid,
                         target_column=col,
                         parameters={"validator": "email", "replacement": None}))

        # ── Phase 6: Business rules ──
        # These are applied unconditionally — environment handles no-ops gracefully
        q(Action(action_type=ActionType.apply_business_rule,
                 parameters={"rule_name": "fix_negative_quantities"}))
        q(Action(action_type=ActionType.apply_business_rule,
                 parameters={"rule_name": "fix_negative_prices"}))
        q(Action(action_type=ActionType.apply_business_rule,
                 parameters={"rule_name": "normalize_currency"}))
        q(Action(action_type=ActionType.apply_business_rule,
                 parameters={"rule_name": "fix_state_country"}))
        q(Action(action_type=ActionType.apply_business_rule,
                 parameters={"rule_name": "fix_total_comp"}))

        # ── Phase 7: Finalize ──
        q(Action(action_type=ActionType.finalize_cleaning))

    def select_action(self, obs: Observation) -> Action:
        """Pop the next action from the queue. Falls back to finalize if empty."""
        if self._actions_queue:
            return self._actions_queue.pop(0)
        return Action(action_type=ActionType.finalize_cleaning)


# ─── Optional LLM agent stub ─────────────────────────────────────────────────

class LLMAgent:
    """
    Stub for an optional LLM-backed agent.
    Only active when OPENAI_API_KEY is set in environment variables.
    Falls back to HeuristicAgent if no key is available.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id   = task_id
        self._api_key  = os.getenv("OPENAI_API_KEY", "")
        self._fallback = HeuristicAgent(task_id)
        self._ready    = bool(self._api_key)

        if self._ready:
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                self._ready = False

    def plan(self, obs: Observation) -> None:
        self._fallback.plan(obs)

    def select_action(self, obs: Observation) -> Action:
        if not self._ready:
            return self._fallback.select_action(obs)
        # LLM integration point — currently delegates to heuristic
        # Future: send obs as JSON to LLM, parse structured Action response
        return self._fallback.select_action(obs)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_agent(task_id: str, prefer_llm: bool = False) -> HeuristicAgent | LLMAgent:
    """
    Return the best available agent.
    - prefer_llm=True and OPENAI_API_KEY set → LLMAgent (with heuristic fallback)
    - Otherwise → HeuristicAgent (always works)
    """
    if prefer_llm:
        return LLMAgent(task_id)
    return HeuristicAgent(task_id)
