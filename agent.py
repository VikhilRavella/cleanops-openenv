# agent.py
"""
agent.py — Heuristic baseline agent for CleanOps OpenEnv.
"""

from __future__ import annotations
import os
from typing import List, Optional

from models import Action, ActionType, Observation


class HeuristicAgent:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._actions_queue: List[Action] = []

    def plan(self, obs: Observation) -> None:
        """Build a deterministic action queue based on the initial observation."""
        schema    = obs.columns          # ← FIXED: was obs.schema
        qr        = obs.quality_report
        self._actions_queue = []

        def q(action: Action):
            self._actions_queue.append(action)

        # Phase 1: Remove duplicates
        q(Action(action_type=ActionType.remove_duplicates, parameters={"keep": "first"}))

        # Phase 2: Fill missing values
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

        # Phase 3: Standardize text
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
            else:
                q(Action(action_type=ActionType.standardize_format,
                         target_column=col,
                         parameters={"format": "strip"}))

        # Phase 4: Normalize dates
        date_cols = [c.name for c in schema if "date" in c.name.lower()]
        for col in date_cols:
            q(Action(action_type=ActionType.normalize_dates,
                     target_column=col,
                     parameters={"output_format": "%Y-%m-%d"}))

        # Phase 5: Fix phones and emails
        for col in [c.name for c in schema if "phone" in c.name.lower()]:
            q(Action(action_type=ActionType.validate_values,
                     target_column=col,
                     parameters={"validator": "phone"}))

        for col in [c.name for c in schema if "email" in c.name.lower()]:
            if qr.invalid_email_count > 0:
                q(Action(action_type=ActionType.replace_invalid,
                         target_column=col,
                         parameters={"validator": "email", "replacement": None}))

        # Phase 6: Business rules
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

        # Phase 7: Finalize
        q(Action(action_type=ActionType.finalize_cleaning))

    def select_action(self, obs: Observation) -> Action:
        if self._actions_queue:
            return self._actions_queue.pop(0)
        return Action(action_type=ActionType.finalize_cleaning)


class LLMAgent:
    def __init__(self, task_id: str) -> None:
        self.task_id   = task_id
        self._fallback = HeuristicAgent(task_id)

    def plan(self, obs: Observation) -> None:
        self._fallback.plan(obs)

    def select_action(self, obs: Observation) -> Action:
        return self._fallback.select_action(obs)


def get_agent(task_id: str, prefer_llm: bool = False) -> HeuristicAgent | LLMAgent:
    if prefer_llm:
        return LLMAgent(task_id)
    return HeuristicAgent(task_id)
