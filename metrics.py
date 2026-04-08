"""
metrics.py — Data quality metrics layer for CleanOps OpenEnv.

Computes before-vs-after metrics used in:
  - reward shaping (environment.py)
  - final grading (graders.py)
  - /baseline API output
  - README demonstration

All functions are pure (no side effects).
"""

from __future__ import annotations
from typing import Any, Dict
import pandas as pd

from cleaning_rules import (
    count_invalid_emails,
    count_invalid_phones,
    count_invalid_dates,
    count_duplicate_rows,
    count_missing,
)


def _has_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


# ─── Per-issue counters ───────────────────────────────────────────────────────

def missing_value_count(df: pd.DataFrame) -> int:
    return count_missing(df)


def duplicate_row_count(df: pd.DataFrame) -> int:
    return count_duplicate_rows(df)


def invalid_email_count(df: pd.DataFrame) -> int:
    for col in ("email", "email_address", "contact_email"):
        if _has_column(df, col):
            return count_invalid_emails(df[col])
    return 0


def invalid_phone_count(df: pd.DataFrame) -> int:
    for col in ("phone", "phone_number", "contact_phone"):
        if _has_column(df, col):
            return count_invalid_phones(df[col])
    return 0


def invalid_date_count(df: pd.DataFrame) -> int:
    """Count across all columns with 'date' in their name."""
    total = 0
    for col in df.columns:
        if "date" in col.lower():
            total += count_invalid_dates(df[col])
    return total


def formatting_inconsistency_count(df: pd.DataFrame) -> int:
    """
    Heuristic: count string cells that have leading/trailing whitespace
    or are not in a consistent case within their column.
    """
    count = 0
    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        # Leading/trailing spaces
        count += int((series != series.str.strip()).sum())
        # Inconsistent case: if >1 unique case variant, flag them
        lower = series.str.lower()
        upper = series.str.upper()
        count += int(((series != lower) & (series != upper) & (series != series.str.title())).sum())
    return count


def preserved_row_ratio(original_df: pd.DataFrame, current_df: pd.DataFrame) -> float:
    """Fraction of original rows still present (after de-dup, not mass deletion)."""
    if len(original_df) == 0:
        return 1.0
    return min(1.0, len(current_df) / len(original_df))


def destructive_change_penalty(original_df: pd.DataFrame, current_df: pd.DataFrame) -> float:
    """
    Returns a penalty [0, 1] for losing too many rows without justification.
    Penalty triggers when >40% of rows are removed.
    """
    ratio = preserved_row_ratio(original_df, current_df)
    if ratio >= 0.6:
        return 0.0
    # Linear penalty from 0 at 60% retention to 1.0 at 0% retention
    return round(1.0 - (ratio / 0.6), 4)


def data_quality_score(df: pd.DataFrame) -> float:
    """
    Composite quality score [0, 1] based on issue density.
    """
    if df.empty:
        return 0.0
    total_cells = df.shape[0] * max(df.shape[1], 1)
    issue_counts = {
        "missing":      missing_value_count(df),
        "duplicates":   duplicate_row_count(df),
        "bad_emails":   invalid_email_count(df),
        "bad_phones":   invalid_phone_count(df),
        "bad_dates":    invalid_date_count(df),
        "formatting":   min(formatting_inconsistency_count(df), total_cells // 2),
    }
    total_issues = sum(issue_counts.values())
    raw = 1.0 - (total_issues / total_cells)
    return round(max(0.0, min(1.0, raw)), 4)


# ─── Full before/after snapshot ───────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a full metrics dict for a single DataFrame state."""
    return {
        "row_count":                  int(len(df)),
        "column_count":               int(len(df.columns)),
        "missing_value_count":        missing_value_count(df),
        "duplicate_row_count":        duplicate_row_count(df),
        "invalid_email_count":        invalid_email_count(df),
        "invalid_phone_count":        invalid_phone_count(df),
        "invalid_date_count":         invalid_date_count(df),
        "formatting_inconsistencies": formatting_inconsistency_count(df),
        "quality_score":              data_quality_score(df),
    }


def compute_before_after(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Return a before-vs-after comparison dict."""
    before = compute_metrics(original_df)
    after  = compute_metrics(cleaned_df)

    improvements: Dict[str, int] = {}
    for key in before:
        if isinstance(before[key], (int, float)) and key != "quality_score":
            improvements[key] = int(before[key]) - int(after[key])

    return {
        "before":       before,
        "after":        after,
        "improvements": improvements,
        "quality_delta": round(after["quality_score"] - before["quality_score"], 4),
        "preservation_ratio": preserved_row_ratio(original_df, cleaned_df),
        "destructive_penalty": destructive_change_penalty(original_df, cleaned_df),
    }
