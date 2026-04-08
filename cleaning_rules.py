"""
cleaning_rules.py — Reusable deterministic cleaning helper functions.

All functions are pure: they take a value (or Series) and return a cleaned
value (or Series). No DataFrame mutation happens here — that belongs to
server/environment.py.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Optional


# ─── Email ───────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$")

def is_valid_email(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return bool(EMAIL_RE.match(value.strip()))


def normalize_email(value: str) -> str:
    """Lowercase and strip whitespace from an email."""
    if not isinstance(value, str):
        return value
    return value.strip().lower()


def count_invalid_emails(series: pd.Series) -> int:
    non_null = series.dropna()
    return int((~non_null.astype(str).apply(is_valid_email)).sum())


# ─── Phone ───────────────────────────────────────────────────────────────────

def normalize_phone(value: str) -> str:
    """
    Strip all non-digit characters and format as (XXX) XXX-XXXX
    for 10-digit US numbers. Returns original if not 10 digits.
    """
    if not isinstance(value, str):
        return value
    digits = re.sub(r"\D", "", value)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == "1":
        d = digits[1:]
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return value  # can't normalize — return as-is


def is_valid_phone(value: str) -> bool:
    if not isinstance(value, str):
        return False
    digits = re.sub(r"\D", "", value)
    return len(digits) in (10, 11)


def count_invalid_phones(series: pd.Series) -> int:
    non_null = series.dropna()
    return int((~non_null.astype(str).apply(is_valid_phone)).sum())


# ─── Dates ───────────────────────────────────────────────────────────────────

COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y",
    "%d-%m-%Y", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
    "%B, %Y", "%b %Y",
]


def parse_date_flexible(value: str) -> Optional[pd.Timestamp]:
    """Try common date formats; return Timestamp or None."""
    if not isinstance(value, str):
        return None
    for fmt in COMMON_DATE_FORMATS:
        try:
            return pd.to_datetime(value.strip(), format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(value.strip(), infer_datetime_format=True)
    except Exception:
        return None


def normalize_date(value: str, output_format: str = "%Y-%m-%d") -> str:
    """Parse and reformat a date string. Returns original on failure."""
    ts = parse_date_flexible(str(value))
    if ts is not None:
        return ts.strftime(output_format)
    return value


def is_valid_date(value: str) -> bool:
    return parse_date_flexible(str(value)) is not None


def count_invalid_dates(series: pd.Series) -> int:
    non_null = series.dropna()
    return int((~non_null.astype(str).apply(is_valid_date)).sum())


# ─── Text / Capitalization ────────────────────────────────────────────────────

def standardize_text(value: str, mode: str = "strip") -> str:
    """
    Apply a text standardization mode to a single value.
    Modes: strip | lower | upper | title_case
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    if mode == "lower":
        return value.lower()
    if mode == "upper":
        return value.upper()
    if mode == "title_case":
        return value.title()
    return value  # strip only


def normalize_series_text(series: pd.Series, mode: str = "strip") -> pd.Series:
    return series.apply(lambda v: standardize_text(v, mode) if isinstance(v, str) else v)


# ─── Duplicates ───────────────────────────────────────────────────────────────

def count_duplicate_rows(df: pd.DataFrame, subset=None) -> int:
    return int(df.duplicated(subset=subset).sum())


def remove_duplicate_rows(df: pd.DataFrame, subset=None, keep: str = "first") -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


# ─── Missing values ───────────────────────────────────────────────────────────

def count_missing(df: pd.DataFrame) -> int:
    return int(df.isnull().sum().sum())


def fill_missing_values(series: pd.Series, strategy: str = "value", value=None) -> pd.Series:
    """
    Fill missing values in a Series.
    Strategies: 'value' (use provided value), 'mode', 'ffill', 'bfill'
    """
    if strategy == "mode":
        mode_val = series.mode()
        fill = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
        return series.fillna(fill)
    if strategy == "ffill":
        return series.ffill()
    if strategy == "bfill":
        return series.bfill()
    return series.fillna(value if value is not None else "UNKNOWN")


# ─── Category normalization ───────────────────────────────────────────────────

CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "status": {
        "active": "Active", "actv": "Active", "act": "Active",
        "inactive": "Inactive", "inact": "Inactive", "in-active": "Inactive",
        "pending": "Pending", "pend": "Pending",
        "closed": "Closed", "close": "Closed",
    },
    "currency": {
        "usd": "USD", "us dollar": "USD", "dollar": "USD", "$": "USD",
        "eur": "EUR", "euro": "EUR", "€": "EUR",
        "gbp": "GBP", "pound": "GBP", "£": "GBP",
    },
}


def normalize_category(value: str, category_type: str) -> str:
    """Map known aliases to canonical category values."""
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    aliases = CATEGORY_ALIASES.get(category_type, {})
    return aliases.get(key, value.strip())


# ─── Numeric / business rules ─────────────────────────────────────────────────

def fix_negative_quantities(series: pd.Series) -> pd.Series:
    """Replace negative numeric values with their absolute values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.abs()


def fix_negative_prices(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.abs()


def fix_invalid_ages(series: pd.Series, min_age: int = 0, max_age: int = 120) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where((numeric >= min_age) & (numeric <= max_age), other=pd.NA)


def normalize_currency_amount(value) -> float | None:
    """Strip currency symbols and parse numeric value."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


# ─── Cross-column business rules ─────────────────────────────────────────────

VALID_STATE_COUNTRY: dict[str, str] = {
    "CA": "US", "NY": "US", "TX": "US", "FL": "US", "WA": "US",
    "ON": "CA", "BC": "CA", "QC": "CA",
}


def is_valid_state_country(state: str, country: str) -> bool:
    if not isinstance(state, str) or not isinstance(country, str):
        return True  # can't validate, give benefit of the doubt
    expected_country = VALID_STATE_COUNTRY.get(state.strip().upper())
    if expected_country is None:
        return True  # unknown state — skip
    return country.strip().upper() == expected_country


def validate_total(row: pd.Series, qty_col: str, price_col: str, total_col: str,
                   tolerance: float = 0.02) -> bool:
    """Check whether total ≈ qty * price within a relative tolerance."""
    try:
        qty   = float(row[qty_col])
        price = float(row[price_col])
        total = float(row[total_col])
        expected = qty * price
        if expected == 0:
            return total == 0
        return abs(total - expected) / abs(expected) <= tolerance
    except (ValueError, TypeError, KeyError):
        return False


# ─── Quality score helper ─────────────────────────────────────────────────────

def compute_quality_score(df: pd.DataFrame, issue_counts: dict) -> float:
    """
    Returns a 0–1 quality score based on issue density.
    Higher = cleaner.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 1.0
    total_issues = sum(issue_counts.values())
    raw = 1.0 - (total_issues / total_cells)
    return round(max(0.0, min(1.0, raw)), 4)
