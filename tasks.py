"""
tasks.py — Benchmark task definitions for CleanOps OpenEnv.

Each task contains:
  - task metadata (id, name, difficulty, description)
  - messy input dataset (as list of dicts → loaded into pandas DataFrame)
  - hidden expected cleaned dataset (used only by grader)
  - expected_issue_types (hints for agents)
  - max_steps

Tasks:
  task_001  EASY    Customer Contact Cleanup
  task_002  MEDIUM  Sales Transaction Repair
  task_003  HARD    Enterprise Ops Multi-Rule Cleanup
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd
import copy


@dataclass
class TaskDefinition:
    task_id: str
    task_name: str
    difficulty: str           # "easy" | "medium" | "hard"
    description: str
    expected_issue_types: List[str]
    max_steps: int
    _messy_rows: List[Dict[str, Any]] = field(repr=False)
    _expected_rows: List[Dict[str, Any]] = field(repr=False)

    def get_messy_df(self) -> pd.DataFrame:
        return pd.DataFrame(copy.deepcopy(self._messy_rows))

    def get_expected_df(self) -> pd.DataFrame:
        return pd.DataFrame(copy.deepcopy(self._expected_rows))

    def to_info_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "description": self.description,
            "expected_issue_types": self.expected_issue_types,
            "max_steps": self.max_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 001 — EASY — Customer Contact Cleanup
# ─────────────────────────────────────────────────────────────────────────────

_TASK_001_MESSY = [
    {"customer_id": "C001", "name": "  alice johnson ",  "email": "Alice.Johnson@Example.COM",  "phone": "555-123-4567",   "city": "new york"},
    {"customer_id": "C002", "name": "BOB SMITH",          "email": "bob.smith@example.com",      "phone": "(212)5554321",   "city": "New York"},
    {"customer_id": "C003", "name": "Carol  White",       "email": "carol.white@",               "phone": "415.555.9876",   "city": "san francisco"},
    {"customer_id": "C004", "name": "david  lee",         "email": "david.lee@company.org",      "phone": "312-555-0011",   "city": "CHICAGO"},
    {"customer_id": "C005", "name": "  Eve Martinez ",    "email": "eve.martinez@biz.net",       "phone": "5551234567",     "city": "miami"},
    {"customer_id": "C006", "name": "Frank Garcia",       "email": "frank.garcia@example.com",   "phone": "555.678.1234",   "city": "Los Angeles"},
    {"customer_id": "C007", "name": "  grace kim  ",      "email": "grace.kim@example.com",      "phone": "(312) 555-7890", "city": "chicago"},
    {"customer_id": "C008", "name": "Hank Wilson",        "email": "hank.wilson@EXAMPLE.COM",    "phone": "7185550001",     "city": "New York"},
    # Duplicate of C001 (same email)
    {"customer_id": "C009", "name": "Alice Johnson",      "email": "alice.johnson@example.com",  "phone": "555-123-4567",   "city": "New York"},
    {"customer_id": "C010", "name": "Irene Patel",        "email": "irene.patel@domain.io",      "phone": "9495559988",     "city": "  Los Angeles  "},
    # Exact duplicate row
    {"customer_id": "C004", "name": "david  lee",         "email": "david.lee@company.org",      "phone": "312-555-0011",   "city": "CHICAGO"},
]

_TASK_001_EXPECTED = [
    {"customer_id": "C001", "name": "Alice Johnson",  "email": "alice.johnson@example.com", "phone": "(555) 123-4567", "city": "New York"},
    {"customer_id": "C002", "name": "Bob Smith",      "email": "bob.smith@example.com",     "phone": "(212) 555-4321", "city": "New York"},
    {"customer_id": "C003", "name": "Carol White",    "email": None,                        "phone": "(415) 555-9876", "city": "San Francisco"},
    {"customer_id": "C004", "name": "David Lee",      "email": "david.lee@company.org",     "phone": "(312) 555-0011", "city": "Chicago"},
    {"customer_id": "C005", "name": "Eve Martinez",   "email": "eve.martinez@biz.net",      "phone": "(555) 123-4567", "city": "Miami"},
    {"customer_id": "C006", "name": "Frank Garcia",   "email": "frank.garcia@example.com",  "phone": "(555) 678-1234", "city": "Los Angeles"},
    {"customer_id": "C007", "name": "Grace Kim",      "email": "grace.kim@example.com",     "phone": "(312) 555-7890", "city": "Chicago"},
    {"customer_id": "C008", "name": "Hank Wilson",    "email": "hank.wilson@example.com",   "phone": "(718) 555-0001", "city": "New York"},
    {"customer_id": "C010", "name": "Irene Patel",    "email": "irene.patel@domain.io",     "phone": "(949) 555-9988", "city": "Los Angeles"},
]

TASK_001 = TaskDefinition(
    task_id="task_001",
    task_name="Customer Contact Cleanup",
    difficulty="easy",
    description=(
        "A CRM export with messy customer contact records. "
        "Issues include bad capitalization, extra whitespace, invalid email addresses, "
        "inconsistent phone formats, duplicate rows, and inconsistent city names. "
        "Clean the dataset so all records are properly formatted and de-duplicated."
    ),
    expected_issue_types=[
        "bad_capitalization", "extra_whitespace", "invalid_email",
        "inconsistent_phone_format", "duplicate_rows", "inconsistent_city",
    ],
    max_steps=15,
    _messy_rows=_TASK_001_MESSY,
    _expected_rows=_TASK_001_EXPECTED,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 002 — MEDIUM — Sales Transaction Repair
# ─────────────────────────────────────────────────────────────────────────────

_TASK_002_MESSY = [
    {"invoice_id": "INV-001", "date": "2024-01-15",    "product": "Widget A",    "category": "Hardware",   "quantity": 5,    "unit_price": 19.99,  "currency": "USD",    "total": 99.95},
    {"invoice_id": "INV-002", "date": "15/01/2024",    "product": "gadget b",    "category": None,         "quantity": -3,   "unit_price": 49.99,  "currency": "usd",    "total": 149.97},
    {"invoice_id": "INV-003", "date": "January 20 2024","product": "Widget A",   "category": "hardware",   "quantity": 2,    "unit_price": 19.99,  "currency": "USD",    "total": 39.98},
    {"invoice_id": "INV-004", "date": "2024-02-01",    "product": "SERVICE PACK","category": "Services",   "quantity": 1,    "unit_price": 250.0,  "currency": "$",      "total": 250.0},
    {"invoice_id": "INV-005", "date": "02-05-2024",    "product": "Gadget B",    "category": "Electronics","quantity": 7,    "unit_price": 49.99,  "currency": "USD",    "total": 349.93},
    {"invoice_id": "INV-006", "date": "2024/03/10",    "product": "widget a",    "category": "HARDWARE",   "quantity": 10,   "unit_price": 19.99,  "currency": "USD",    "total": 199.90},
    # Duplicate invoice
    {"invoice_id": "INV-004", "date": "2024-02-01",    "product": "SERVICE PACK","category": "Services",   "quantity": 1,    "unit_price": 250.0,  "currency": "$",      "total": 250.0},
    {"invoice_id": "INV-007", "date": "not-a-date",    "product": "Accessory C", "category": "Accessories","quantity": 4,    "unit_price": 12.50,  "currency": "USD",    "total": 50.0},
    {"invoice_id": "INV-008", "date": "2024-03-22",    "product": "Gadget B",    "category": "Electronics","quantity": -1,   "unit_price": 49.99,  "currency": "EUR",    "total": -49.99},
    {"invoice_id": "INV-009", "date": "2024-04-01",    "product": "Accessory C", "category": "accessories","quantity": 6,    "unit_price": 12.50,  "currency": "USD",    "total": 75.0},
    {"invoice_id": "INV-010", "date": "2024/04/15",    "product": None,          "category": "Hardware",   "quantity": 3,    "unit_price": 19.99,  "currency": "USD",    "total": 59.97},
]

_TASK_002_EXPECTED = [
    {"invoice_id": "INV-001", "date": "2024-01-15", "product": "Widget A",    "category": "Hardware",    "quantity": 5, "unit_price": 19.99, "currency": "USD", "total": 99.95},
    {"invoice_id": "INV-002", "date": "2024-01-15", "product": "Gadget B",    "category": "Electronics", "quantity": 3, "unit_price": 49.99, "currency": "USD", "total": 149.97},
    {"invoice_id": "INV-003", "date": "2024-01-20", "product": "Widget A",    "category": "Hardware",    "quantity": 2, "unit_price": 19.99, "currency": "USD", "total": 39.98},
    {"invoice_id": "INV-004", "date": "2024-02-01", "product": "Service Pack", "category": "Services",   "quantity": 1, "unit_price": 250.0, "currency": "USD", "total": 250.0},
    {"invoice_id": "INV-005", "date": "2024-02-05", "product": "Gadget B",    "category": "Electronics", "quantity": 7, "unit_price": 49.99, "currency": "USD", "total": 349.93},
    {"invoice_id": "INV-006", "date": "2024-03-10", "product": "Widget A",    "category": "Hardware",    "quantity": 10,"unit_price": 19.99, "currency": "USD", "total": 199.90},
    {"invoice_id": "INV-008", "date": "2024-03-22", "product": "Gadget B",    "category": "Electronics", "quantity": 1, "unit_price": 49.99, "currency": "EUR", "total": 49.99},
    {"invoice_id": "INV-009", "date": "2024-04-01", "product": "Accessory C", "category": "Accessories", "quantity": 6, "unit_price": 12.50, "currency": "USD", "total": 75.0},
    {"invoice_id": "INV-010", "date": "2024-04-15", "product": "UNKNOWN",     "category": "Hardware",    "quantity": 3, "unit_price": 19.99, "currency": "USD", "total": 59.97},
]

TASK_002 = TaskDefinition(
    task_id="task_002",
    task_name="Sales Transaction Repair",
    difficulty="medium",
    description=(
        "A sales/invoice export with multiple data quality issues. "
        "Issues include mixed date formats, negative quantities, missing categories, "
        "duplicate invoice rows, inconsistent currency formatting, and non-title-case product names. "
        "Repair the transaction records while preserving valid data."
    ),
    expected_issue_types=[
        "mixed_date_formats", "negative_quantities", "missing_category",
        "duplicate_invoice", "inconsistent_currency", "inconsistent_product_names",
        "missing_product",
    ],
    max_steps=20,
    _messy_rows=_TASK_002_MESSY,
    _expected_rows=_TASK_002_EXPECTED,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 003 — HARD — Enterprise Ops Multi-Rule Cleanup
# ─────────────────────────────────────────────────────────────────────────────

_TASK_003_MESSY = [
    {"record_id": "R001", "employee_id": "E-1001",  "name": "sarah  Connor",   "department": "Engineering", "hire_date": "2020-03-15",  "state": "CA", "country": "US",  "salary": 95000, "bonus_pct": 10, "total_comp": 104500},
    {"record_id": "R002", "employee_id": None,       "name": "John  Doe",       "department": "marketing",   "hire_date": "19/06/2021",  "state": "NY", "country": "US",  "salary": 72000, "bonus_pct": 8,  "total_comp": 77760},
    {"record_id": "R003", "employee_id": "E-1003",  "name": "MIKE CHEN",        "department": "Engineering", "hire_date": "2021-07-01",  "state": "TX", "country": "CA",  "salary": 88000, "bonus_pct": 10, "total_comp": 96800},
    {"record_id": "R004", "employee_id": "E-1004",  "name": "Lisa Park",        "department": "Finance",     "hire_date": "March 2022",  "state": "WA", "country": "US",  "salary": 81000, "bonus_pct": 7,  "total_comp": 86670},
    {"record_id": "R005", "employee_id": "E-1005",  "name": "carlos  Rivera",   "department": "SALES",       "hire_date": "2022/10/01",  "state": "FL", "country": "US",  "salary": 65000, "bonus_pct": 15, "total_comp": 74750},
    {"record_id": "R006", "employee_id": "E-1006",  "name": "Angela White",     "department": "Engineering", "hire_date": "2023-01-10",  "state": "CA", "country": "US",  "salary": 102000,"bonus_pct": 10, "total_comp": 99000},  # wrong total
    {"record_id": "R007", "employee_id": "E-1007",  "name": "  Tom  Harris  ",  "department": "HR",          "hire_date": "01/15/2023",  "state": "ON", "country": "US",  "salary": 58000, "bonus_pct": 5,  "total_comp": 60900},  # ON is Canada, not US
    {"record_id": "R008", "employee_id": "E-1008",  "name": "Priya Nair",       "department": "finance",     "hire_date": "2023-06-01",  "state": "NY", "country": "US",  "salary": 83000, "bonus_pct": 7,  "total_comp": 88810},  # wrong total
    # Ambiguous duplicate — same employee, different record_id
    {"record_id": "R009", "employee_id": "E-1001",  "name": "Sarah Connor",     "department": "Engineering", "hire_date": "2020-03-15",  "state": "CA", "country": "US",  "salary": 95000, "bonus_pct": 10, "total_comp": 104500},
    {"record_id": "R010", "employee_id": "E-1010",  "name": "Wei  Zhang",       "department": "Engineering", "hire_date": "2023-11-01",  "state": "BC", "country": "CA",  "salary": 91000, "bonus_pct": 10, "total_comp": 100100},
    {"record_id": "R011", "employee_id": "E-1011",  "name": "nina  Torres",     "department": "Sales",       "hire_date": "2024/01/15",  "state": "TX", "country": "US",  "salary": 70000, "bonus_pct": 15, "total_comp": 80500},
]

_TASK_003_EXPECTED = [
    {"record_id": "R001", "employee_id": "E-1001", "name": "Sarah Connor",   "department": "Engineering", "hire_date": "2020-03-15", "state": "CA", "country": "US", "salary": 95000, "bonus_pct": 10, "total_comp": 104500},
    {"record_id": "R002", "employee_id": "MISSING","name": "John Doe",       "department": "Marketing",   "hire_date": "2021-06-19", "state": "NY", "country": "US", "salary": 72000, "bonus_pct": 8,  "total_comp": 77760},
    {"record_id": "R003", "employee_id": "E-1003", "name": "Mike Chen",      "department": "Engineering", "hire_date": "2021-07-01", "state": "TX", "country": "US", "salary": 88000, "bonus_pct": 10, "total_comp": 96800},
    {"record_id": "R004", "employee_id": "E-1004", "name": "Lisa Park",      "department": "Finance",     "hire_date": "2022-03-01", "state": "WA", "country": "US", "salary": 81000, "bonus_pct": 7,  "total_comp": 86670},
    {"record_id": "R005", "employee_id": "E-1005", "name": "Carlos Rivera",  "department": "Sales",       "hire_date": "2022-10-01", "state": "FL", "country": "US", "salary": 65000, "bonus_pct": 15, "total_comp": 74750},
    {"record_id": "R006", "employee_id": "E-1006", "name": "Angela White",   "department": "Engineering", "hire_date": "2023-01-10", "state": "CA", "country": "US", "salary": 102000,"bonus_pct": 10, "total_comp": 112200},
    {"record_id": "R007", "employee_id": "E-1007", "name": "Tom Harris",     "department": "HR",          "hire_date": "2023-01-15", "state": "ON", "country": "CA", "salary": 58000, "bonus_pct": 5,  "total_comp": 60900},
    {"record_id": "R008", "employee_id": "E-1008", "name": "Priya Nair",     "department": "Finance",     "hire_date": "2023-06-01", "state": "NY", "country": "US", "salary": 83000, "bonus_pct": 7,  "total_comp": 88810},
    {"record_id": "R010", "employee_id": "E-1010", "name": "Wei Zhang",      "department": "Engineering", "hire_date": "2023-11-01", "state": "BC", "country": "CA", "salary": 91000, "bonus_pct": 10, "total_comp": 100100},
    {"record_id": "R011", "employee_id": "E-1011", "name": "Nina Torres",    "department": "Sales",       "hire_date": "2024-01-15", "state": "TX", "country": "US", "salary": 70000, "bonus_pct": 15, "total_comp": 80500},
]

TASK_003 = TaskDefinition(
    task_id="task_003",
    task_name="Enterprise Ops Multi-Rule Cleanup",
    difficulty="hard",
    description=(
        "An enterprise HR / ops reporting table with complex, interrelated data issues. "
        "Issues include mixed date formats, missing required employee IDs, "
        "cross-column inconsistencies (state/country mismatch), ambiguous duplicate records, "
        "incorrect total_comp calculations, bad capitalization, and business policy violations. "
        "Cleaning requires applying multiple business rules in a logical order."
    ),
    expected_issue_types=[
        "missing_required_id", "mixed_date_formats", "bad_capitalization",
        "extra_whitespace", "state_country_mismatch", "incorrect_total_comp",
        "ambiguous_duplicates", "inconsistent_department_names",
    ],
    max_steps=25,
    _messy_rows=_TASK_003_MESSY,
    _expected_rows=_TASK_003_EXPECTED,
)


# ─── Registry ─────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "task_001": TASK_001,
    "task_002": TASK_002,
    "task_003": TASK_003,
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[Dict[str, Any]]:
    return [t.to_info_dict() for t in TASK_REGISTRY.values()]
