from __future__ import annotations



from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from models import (
    Action, ActionType, Observation, State,
    ColumnInfo, QualityReport,
)
from tasks import get_task, TaskDefinition
from metrics import compute_metrics, data_quality_score
import cleaning_rules as cr


class CleanOpsEnvironment:

    def __init__(self) -> None:
        self._task: Optional[TaskDefinition] = None
        self._original_df: Optional[pd.DataFrame] = None
        self._current_df: Optional[pd.DataFrame] = None
        self._expected_df: Optional[pd.DataFrame] = None
        self._action_history: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._is_done: bool = False
        self._last_reward: float = 0.0
        self._score_before_step: float = 0.0

    def reset(self, task_id: str) -> Observation:
        self._task        = get_task(task_id)
        self._original_df = self._task.get_messy_df()
        self._current_df  = self._task.get_messy_df()
        self._expected_df = self._task.get_expected_df()
        self._action_history = []
        self._step_count  = 0
        self._is_done     = False
        self._last_reward = 0.0
        self._score_before_step = data_quality_score(self._current_df)
        return self._build_observation(
            reward=0.0, done=False,
            message=f"Episode started. Task: {self._task.task_name}. "
                    f"Difficulty: {self._task.difficulty.upper()}. "
                    f"Max steps: {self._task.max_steps}.",
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if not self._task:
            # Auto-reset to task_001 if called before reset
            self.reset("task_001")

        if self._is_done:
            return self._build_observation(0.0, True, "Episode already finished."), 0.0, True, {}

        self._step_count += 1
        info: Dict[str, Any] = {}

        try:
            reward, message = self._dispatch(action)
        except Exception as exc:
            reward  = -0.05
            message = f"Action failed: {exc}"

        self._action_history.append({
            "step":          self._step_count,
            "action_type":   action.action_type,
            "target_column": action.target_column,
            "parameters":    action.parameters,
            "message":       message,
            "reward":        reward,
        })

        done = (
            action.action_type == ActionType.finalize_cleaning
            or self._step_count >= self._task.max_steps
        )
        if done:
            self._is_done = True
            if self._step_count >= self._task.max_steps and action.action_type != ActionType.finalize_cleaning:
                reward -= 0.10
                message += " [Max steps reached — episode auto-terminated.]"

        self._last_reward = reward
        obs = self._build_observation(reward=reward, done=done, message=message)
        return obs, reward, done, info

    def state(self) -> State:
        if not self._task:
            raise RuntimeError("Call reset() first.")
        return State(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            original_df=self._original_df.where(pd.notnull(self._original_df), None).to_dict("records"),
            current_df=self._current_df.where(pd.notnull(self._current_df), None).to_dict("records"),
            expected_df=self._expected_df.where(pd.notnull(self._expected_df), None).to_dict("records"),
            issue_summary=compute_metrics(self._current_df),
            action_history=self._action_history,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            is_done=self._is_done,
            current_score_estimate=data_quality_score(self._current_df),
        )

    def _dispatch(self, action: Action) -> Tuple[float, str]:
        df     = self._current_df
        at     = action.action_type
        col    = action.target_column
        params = action.parameters

        # ── Safety guard: column existence check ──
        mutation_actions = {
            ActionType.standardize_format, ActionType.normalize_dates,
            ActionType.validate_values, ActionType.replace_invalid,
            ActionType.fill_missing,
        }
        if at in mutation_actions:
            if not col:
                return -0.05, f"Action '{at}' requires a target_column."
            if col not in df.columns:
                return -0.05, (
                    f"Column '{col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        if at == ActionType.inspect_column:
            return self._act_inspect(df, col)
        if at == ActionType.standardize_format:
            return self._act_standardize(df, col, params)
        if at == ActionType.normalize_dates:
            return self._act_normalize_dates(df, col, params)
        if at == ActionType.validate_values:
            return self._act_validate(df, col, params)
        if at == ActionType.replace_invalid:
            return self._act_replace_invalid(df, col, params)
        if at == ActionType.fill_missing:
            return self._act_fill_missing(df, col, params)
        if at == ActionType.remove_duplicates:
            return self._act_remove_duplicates(df, params)
        if at == ActionType.merge_duplicates:
            return self._act_merge_duplicates(df, params)
        if at == ActionType.apply_business_rule:
            return self._act_business_rule(df, params)
        if at == ActionType.finalize_cleaning:
            return self._act_finalize(df)

        return -0.05, f"Unknown action type: {at}"

    def _act_inspect(self, df: pd.DataFrame, col: Optional[str]) -> Tuple[float, str]:
        if col and col not in df.columns:
            return -0.02, f"Column '{col}' not found. Available: {list(df.columns)}"
        return 0.01, f"Inspected {'column: ' + col if col else 'full table'}."

    def _act_standardize(self, df: pd.DataFrame, col: str, params: Dict) -> Tuple[float, str]:
        mode   = params.get("format", "strip")
        before = df[col].copy()
        df[col] = cr.normalize_series_text(df[col], mode=mode)
        changed = int((df[col] != before).sum())
        if changed == 0:
            return -0.02, f"No changes in '{col}' (already clean)."
        return self._reward_for_changes(changed, df, f"Standardized {changed} values in '{col}' to {mode}.")

    def _act_normalize_dates(self, df: pd.DataFrame, col: str, params: Dict) -> Tuple[float, str]:
        out_fmt = params.get("output_format", "%Y-%m-%d")
        before  = df[col].copy()
        df[col] = df[col].apply(lambda v: cr.normalize_date(str(v), out_fmt) if pd.notna(v) else v)
        changed = int((df[col] != before).sum())
        if changed == 0:
            return -0.01, f"No date changes in '{col}'."
        return self._reward_for_changes(changed, df, f"Normalized {changed} dates in '{col}'.")

    def _act_validate(self, df: pd.DataFrame, col: str, params: Dict) -> Tuple[float, str]:
        validator = params.get("validator", "")
        if validator == "email":
            count = cr.count_invalid_emails(df[col])
            return 0.02, f"Validated '{col}': {count} invalid email(s) found."
        if validator == "phone":
            count = cr.count_invalid_phones(df[col])
            return 0.02, f"Validated '{col}': {count} invalid phone(s) found."
        if validator == "date":
            count = cr.count_invalid_dates(df[col])
            return 0.02, f"Validated '{col}': {count} invalid date(s) found."
        return 0.01, f"Validated column '{col}'."

    def _act_replace_invalid(self, df: pd.DataFrame, col: str, params: Dict) -> Tuple[float, str]:
        replacement = params.get("replacement", None)
        validator   = params.get("validator", "")
        before_null = df[col].isna().sum()

        if validator == "email":
            mask = ~df[col].apply(cr.is_valid_email)
            df.loc[mask & df[col].notna(), col] = replacement
        elif validator == "phone":
            mask = ~df[col].apply(lambda v: cr.is_valid_phone(str(v)) if pd.notna(v) else True)
            df.loc[mask, col] = replacement
        else:
            return -0.02, "No validator specified for replace_invalid."

        after_null = df[col].isna().sum()
        changed    = int(abs(after_null - before_null))
        if replacement is not None:
            changed = max(changed, int((df[col] == replacement).sum()))
        if changed == 0:
            return -0.02, f"No invalid values found in '{col}'."
        return self._reward_for_changes(changed, df, f"Replaced {changed} invalid '{validator}' values in '{col}'.")

    def _act_fill_missing(self, df: pd.DataFrame, col: str, params: Dict) -> Tuple[float, str]:
        before_nulls = int(df[col].isna().sum())
        if before_nulls == 0:
            return -0.02, f"No missing values in '{col}'."
        strategy = params.get("strategy", "value")
        fill_val  = params.get("value", "UNKNOWN")
        df[col]  = cr.fill_missing_values(df[col], strategy=strategy, value=fill_val)
        filled   = before_nulls - int(df[col].isna().sum())
        return self._reward_for_changes(filled, df, f"Filled {filled} missing values in '{col}'.")

    def _act_remove_duplicates(self, df: pd.DataFrame, params: Dict) -> Tuple[float, str]:
        before = len(df)
        subset = params.get("subset", None)
        keep   = params.get("keep", "first")
        self._current_df = cr.remove_duplicate_rows(df, subset=subset, keep=keep)
        removed = before - len(self._current_df)
        if removed == 0:
            return -0.02, "No duplicate rows found."
        return 0.10 + 0.02 * min(removed, 5), f"Removed {removed} duplicate row(s)."

    def _act_merge_duplicates(self, df: pd.DataFrame, params: Dict) -> Tuple[float, str]:
        key_col = params.get("key_column", None)
        if not key_col or key_col not in df.columns:
            return -0.05, f"Invalid key_column '{key_col}'. Available: {list(df.columns)}"
        before = len(df)
        self._current_df = df.drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)
        removed = before - len(self._current_df)
        if removed == 0:
            return -0.02, f"No duplicates found on key '{key_col}'."
        return 0.10 + 0.02 * min(removed, 5), f"Merged {removed} duplicate(s) on '{key_col}'."

    def _act_business_rule(self, df: pd.DataFrame, params: Dict) -> Tuple[float, str]:
        rule = params.get("rule_name", "")

        if rule == "fix_negative_quantities":
            if "quantity" not in df.columns:
                return -0.02, "Column 'quantity' not found — skipping."
            neg = int((pd.to_numeric(df["quantity"], errors="coerce") < 0).sum())
            if neg == 0:
                return -0.02, "No negative quantities found."
            df["quantity"] = cr.fix_negative_quantities(df["quantity"])
            self._current_df = df
            return 0.10 + 0.02 * neg, f"Fixed {neg} negative quantity value(s)."

        if rule == "fix_negative_prices":
            if "unit_price" not in df.columns:
                return -0.02, "Column 'unit_price' not found — skipping."
            neg = int((pd.to_numeric(df["unit_price"], errors="coerce") < 0).sum())
            if neg == 0:
                return -0.02, "No negative prices found."
            df["unit_price"] = cr.fix_negative_prices(df["unit_price"])
            self._current_df = df
            return 0.10 + 0.02 * neg, f"Fixed {neg} negative price(s)."

        if rule == "normalize_currency":
            if "currency" not in df.columns:
                return -0.02, "Column 'currency' not found — skipping."
            before = df["currency"].copy()
            df["currency"] = df["currency"].apply(
                lambda v: cr.normalize_category(str(v), "currency") if pd.notna(v) else v
            )
            changed = int((df["currency"] != before).sum())
            self._current_df = df
            return (0.08 if changed > 0 else -0.02), f"Normalized {changed} currency value(s)."

        if rule == "fix_missing_employee_ids":
            if "employee_id" not in df.columns:
                return -0.02, "Column 'employee_id' not found — skipping."
            null_mask = df["employee_id"].isna()
            if null_mask.sum() == 0:
                return -0.02, "No missing employee IDs."
            df.loc[null_mask, "employee_id"] = "MISSING"
            self._current_df = df
            return 0.08, f"Flagged {int(null_mask.sum())} missing employee ID(s) as 'MISSING'."

        if rule == "fix_total_comp":
            required = {"salary", "bonus_pct", "total_comp"}
            if not required.issubset(df.columns):
                return -0.02, f"Columns required: {required} — skipping."
            wrong = 0
            for idx, row in df.iterrows():
                try:
                    expected_total = float(row["salary"]) * (1 + float(row["bonus_pct"]) / 100)
                    actual_total   = float(row["total_comp"])
                    if abs(actual_total - expected_total) / max(abs(expected_total), 1) > 0.01:
                        df.at[idx, "total_comp"] = round(expected_total, 2)
                        wrong += 1
                except (ValueError, TypeError):
                    pass
            self._current_df = df
            return (0.12 if wrong > 0 else -0.02), f"Corrected {wrong} total_comp value(s)."

        if rule == "fix_state_country":
            if "state" not in df.columns or "country" not in df.columns:
                return -0.02, "Columns 'state'/'country' not found — skipping."
            fixed = 0
            for idx, row in df.iterrows():
                state   = str(row["state"]).strip().upper()   if pd.notna(row["state"])   else ""
                country = str(row["country"]).strip().upper() if pd.notna(row["country"]) else ""
                expected_country = cr.VALID_STATE_COUNTRY.get(state)
                if expected_country and country != expected_country:
                    df.at[idx, "country"] = expected_country
                    fixed += 1
            self._current_df = df
            return (0.10 if fixed > 0 else -0.02), f"Fixed {fixed} state/country mismatch(es)."

        # Unknown rule — soft fail, no crash
        return -0.02, f"Unknown business rule '{rule}' — skipping."

    def _act_finalize(self, df: pd.DataFrame) -> Tuple[float, str]:
        score_now   = data_quality_score(df)
        score_start = data_quality_score(self._original_df)
        improvement = score_now - score_start
        if improvement >= 0.20:
            bonus, quality = 0.20, "significant"
        elif improvement >= 0.10:
            bonus, quality = 0.10, "moderate"
        elif improvement > 0.0:
            bonus, quality = 0.05, "minor"
        else:
            bonus, quality = -0.10, "no"
        return bonus, (
            f"Cleaning finalized with {quality} improvement "
            f"(quality score: {score_start:.3f} → {score_now:.3f})."
        )

    def _reward_for_changes(self, num_changes: int, df: pd.DataFrame, message: str) -> Tuple[float, str]:
        score_after   = data_quality_score(df)
        delta         = score_after - self._score_before_step
        self._score_before_step = score_after
        reward = round(0.05 + 0.02 * min(num_changes, 10) + delta * 0.5, 4)
        return reward, message

    def _build_observation(self, reward: float, done: bool, message: str) -> Observation:
        df = self._current_df
        m  = compute_metrics(df)
        columns = [
            ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(df[col].isna().sum()),
                unique_count=int(df[col].nunique(dropna=True)),
                sample_values=df[col].dropna().head(3).tolist(),
            )
            for col in df.columns
        ]
        qr = QualityReport(
            total_rows=m["row_count"],
            duplicate_rows=m["duplicate_row_count"],
            missing_value_count=m["missing_value_count"],
            invalid_email_count=m["invalid_email_count"],
            invalid_phone_count=m["invalid_phone_count"],
            invalid_date_count=m["invalid_date_count"],
            invalid_value_count=0,
            quality_score=m["quality_score"],
        )
        preview = df.head(5).where(pd.notnull(df), None).to_dict("records")
        return Observation(
            task_id=self._task.task_id,
            task_name=self._task.task_name,
            task_description=self._task.description,
            difficulty=self._task.difficulty,
            table_preview=preview,
            columns=columns,          # ← renamed from schema (fixes UserWarning)
            quality_report=qr,
            allowed_actions=[a.value for a in ActionType],
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            message=message,
            reward=reward,
            done=done,
        )

    @property
    def current_df(self) -> Optional[pd.DataFrame]:
        return self._current_df

    @property
    def expected_df(self) -> Optional[pd.DataFrame]:
        return self._expected_df

    @property
    def is_done(self) -> bool:
        return self._is_done
