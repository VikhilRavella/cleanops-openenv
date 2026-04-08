"""
graders.py — Deterministic final graders for CleanOps OpenEnv.

Used ONLY after episode completion to compare:
    current_df  (agent's cleaned output)
vs
    expected_df (hidden ground-truth target)

Returns scores in [0.0, 1.0] with a breakdown dict.

Design principles:
  - Pure functions (no side effects, no state mutation)
  - Deterministic (same inputs → same output, always)
  - Partial credit (column-level scoring, not all-or-nothing)
  - Explainable (breakdown dict shows exactly what scored what)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import math

from models import GraderResponse


# ─── Column-level scoring helpers ─────────────────────────────────────────────

def _normalize_str(value) -> str:
    """Lowercase + strip for loose string comparison."""
    if pd.isna(value) or value is None:
        return "__null__"
    return str(value).strip().lower()


def _column_match_score(
    cleaned: pd.Series,
    expected: pd.Series,
    loose: bool = False,
) -> float:
    """
    Fraction of values in `cleaned` that match `expected` (position-aligned).
    If loose=True, uses case-insensitive strip comparison.
    """
    if len(cleaned) == 0 or len(expected) == 0:
        return 0.0
    # Align by position up to min length
    n = min(len(cleaned), len(expected))
    cleaned_vals  = cleaned.reset_index(drop=True).iloc[:n]
    expected_vals = expected.reset_index(drop=True).iloc[:n]

    if loose:
        matches = sum(
            1 for c, e in zip(cleaned_vals, expected_vals)
            if _normalize_str(c) == _normalize_str(e)
        )
    else:
        matches = int((cleaned_vals == expected_vals).sum())

    # Length penalty: if agent returned different row count, penalise
    length_penalty = min(1.0, n / max(len(expected), 1))
    return round((matches / max(n, 1)) * length_penalty, 4)


def _row_count_score(cleaned_len: int, expected_len: int) -> float:
    """Score for matching expected row count. Full credit for exact match."""
    if expected_len == 0:
        return 1.0
    ratio = cleaned_len / expected_len
    if ratio == 1.0:
        return 1.0
    # Partial credit: smooth decay from ±20% deviation
    deviation = abs(ratio - 1.0)
    return round(max(0.0, 1.0 - deviation * 2), 4)


def _null_handling_score(cleaned: pd.Series, expected: pd.Series) -> float:
    """How accurately nulls were placed (matching expected null positions)."""
    n = min(len(cleaned), len(expected))
    if n == 0:
        return 1.0
    c = cleaned.reset_index(drop=True).iloc[:n].isna()
    e = expected.reset_index(drop=True).iloc[:n].isna()
    return round(float((c == e).mean()), 4)


# ─── Task-specific graders ─────────────────────────────────────────────────────

def grade_task_001(
    cleaned_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Grade: Customer Contact Cleanup (EASY)
    Breakdown:
      - row_count_score      (15%)
      - name_score           (20%)  loose
      - email_score          (20%)  strict
      - phone_score          (20%)  strict
      - city_score           (15%)  loose
      - null_handling_score  (10%)
    """
    breakdown: Dict[str, float] = {}

    breakdown["row_count"]     = _row_count_score(len(cleaned_df), len(expected_df))
    breakdown["name"]          = _column_match_score(cleaned_df.get("name", pd.Series([])),     expected_df["name"],  loose=True)
    breakdown["email"]         = _column_match_score(cleaned_df.get("email", pd.Series([])),    expected_df["email"], loose=True)
    breakdown["phone"]         = _column_match_score(cleaned_df.get("phone", pd.Series([])),    expected_df["phone"], loose=False)
    breakdown["city"]          = _column_match_score(cleaned_df.get("city", pd.Series([])),     expected_df["city"],  loose=True)
    breakdown["null_handling"] = _null_handling_score(cleaned_df.get("email", pd.Series([])),   expected_df["email"])

    weights = {
        "row_count":     0.15,
        "name":          0.20,
        "email":         0.20,
        "phone":         0.20,
        "city":          0.15,
        "null_handling": 0.10,
    }
    final = sum(breakdown[k] * weights[k] for k in weights)
    return round(final, 4), breakdown


def grade_task_002(
    cleaned_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Grade: Sales Transaction Repair (MEDIUM)
    Breakdown:
      - row_count_score  (10%)
      - date_score       (20%)  strict ISO
      - product_score    (15%)  loose
      - category_score   (15%)  loose
      - quantity_score   (15%)  strict (negatives fixed)
      - currency_score   (10%)  strict
      - total_score      (15%)  numeric ±2%
    """
    breakdown: Dict[str, float] = {}

    breakdown["row_count"] = _row_count_score(len(cleaned_df), len(expected_df))
    breakdown["date"]      = _column_match_score(cleaned_df.get("date",     pd.Series([])), expected_df["date"],     loose=False)
    breakdown["product"]   = _column_match_score(cleaned_df.get("product",  pd.Series([])), expected_df["product"],  loose=True)
    breakdown["category"]  = _column_match_score(cleaned_df.get("category", pd.Series([])), expected_df["category"], loose=True)
    breakdown["currency"]  = _column_match_score(cleaned_df.get("currency", pd.Series([])), expected_df["currency"], loose=True)

    # Quantity: must be positive
    if "quantity" in cleaned_df.columns and "quantity" in expected_df.columns:
        n = min(len(cleaned_df), len(expected_df))
        cq = pd.to_numeric(cleaned_df["quantity"].reset_index(drop=True).iloc[:n], errors="coerce")
        eq = pd.to_numeric(expected_df["quantity"].reset_index(drop=True).iloc[:n], errors="coerce")
        qty_match = ((cq - eq).abs() < 0.01).sum() / max(n, 1)
        breakdown["quantity"] = round(float(qty_match), 4)
    else:
        breakdown["quantity"] = 0.0

    # Total: within 2% tolerance
    if "total" in cleaned_df.columns and "total" in expected_df.columns:
        n = min(len(cleaned_df), len(expected_df))
        ct = pd.to_numeric(cleaned_df["total"].reset_index(drop=True).iloc[:n], errors="coerce")
        et = pd.to_numeric(expected_df["total"].reset_index(drop=True).iloc[:n], errors="coerce")
        def _within_tol(c, e):
            if pd.isna(c) or pd.isna(e):
                return False
            if e == 0:
                return c == 0
            return abs(c - e) / abs(e) <= 0.02
        total_match = sum(_within_tol(c, e) for c, e in zip(ct, et)) / max(n, 1)
        breakdown["total"] = round(float(total_match), 4)
    else:
        breakdown["total"] = 0.0

    weights = {
        "row_count": 0.10,
        "date":      0.20,
        "product":   0.15,
        "category":  0.15,
        "quantity":  0.15,
        "currency":  0.10,
        "total":     0.15,
    }
    final = sum(breakdown[k] * weights[k] for k in weights)
    return round(final, 4), breakdown


def grade_task_003(
    cleaned_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Grade: Enterprise Ops Multi-Rule Cleanup (HARD)
    Breakdown:
      - row_count_score      (10%)
      - employee_id_score    (15%)  missing → 'MISSING'
      - name_score           (10%)  loose title case
      - department_score     (10%)  loose
      - date_score           (15%)  strict ISO
      - country_score        (15%)  state/country consistency
      - total_comp_score     (15%)  numeric ±1%
      - null_handling_score  (10%)
    """
    breakdown: Dict[str, float] = {}

    breakdown["row_count"]   = _row_count_score(len(cleaned_df), len(expected_df))
    breakdown["employee_id"] = _column_match_score(cleaned_df.get("employee_id", pd.Series([])), expected_df["employee_id"], loose=True)
    breakdown["name"]        = _column_match_score(cleaned_df.get("name",        pd.Series([])), expected_df["name"],        loose=True)
    breakdown["department"]  = _column_match_score(cleaned_df.get("department",  pd.Series([])), expected_df["department"],  loose=True)
    breakdown["hire_date"]   = _column_match_score(cleaned_df.get("hire_date",   pd.Series([])), expected_df["hire_date"],   loose=False)
    breakdown["country"]     = _column_match_score(cleaned_df.get("country",     pd.Series([])), expected_df["country"],     loose=True)

    # total_comp: must match within 1%
    if "total_comp" in cleaned_df.columns and "total_comp" in expected_df.columns:
        n = min(len(cleaned_df), len(expected_df))
        ct = pd.to_numeric(cleaned_df["total_comp"].reset_index(drop=True).iloc[:n], errors="coerce")
        et = pd.to_numeric(expected_df["total_comp"].reset_index(drop=True).iloc[:n], errors="coerce")
        def _comp_match(c, e):
            if pd.isna(c) or pd.isna(e):
                return False
            if e == 0:
                return c == 0
            return abs(c - e) / abs(e) <= 0.01
        comp_match = sum(_comp_match(c, e) for c, e in zip(ct, et)) / max(n, 1)
        breakdown["total_comp"] = round(float(comp_match), 4)
    else:
        breakdown["total_comp"] = 0.0

    breakdown["null_handling"] = _null_handling_score(
        cleaned_df.get("employee_id", pd.Series([])),
        expected_df["employee_id"],
    )

    weights = {
        "row_count":    0.10,
        "employee_id":  0.15,
        "name":         0.10,
        "department":   0.10,
        "hire_date":    0.15,
        "country":      0.15,
        "total_comp":   0.15,
        "null_handling":0.10,
    }
    final = sum(breakdown[k] * weights[k] for k in weights)
    return round(final, 4), breakdown


# ─── Router ───────────────────────────────────────────────────────────────────

_GRADER_MAP = {
    "task_001": grade_task_001,
    "task_002": grade_task_002,
    "task_003": grade_task_003,
}


def grade(
    task_id: str,
    cleaned_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> GraderResponse:
    """
    Main grader entry point.
    Returns GraderResponse with score [0,1] and breakdown.
    """
    if task_id not in _GRADER_MAP:
        raise ValueError(f"No grader for task_id '{task_id}'")

    grader_fn = _GRADER_MAP[task_id]
    score, breakdown = grader_fn(cleaned_df, expected_df)

    if score >= 0.90:
        message = "Excellent cleaning — nearly perfect result!"
    elif score >= 0.70:
        message = "Good cleaning — most issues resolved with minor gaps."
    elif score >= 0.50:
        message = "Partial cleaning — several issues remain."
    else:
        message = "Needs more work — many issues unresolved."

    return GraderResponse(
        task_id=task_id,
        score=score,
        breakdown=breakdown,
        message=message,
    )
