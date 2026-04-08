# CleanOps OpenEnv

> A multi-task AI agent environment for solving real-world data cleaning, normalization, validation, and quality repair problems across messy business datasets.

---

## Why This Project Matters

Real-world data is messy. CRM exports have inconsistent phone formats. Invoice tables have negative quantities. HR reports have cross-column policy violations. Cleaning this data reliably is one of the highest-value, highest-effort tasks in enterprise AI.

CleanOps OpenEnv turns data cleaning into a **structured, gradable, agentic benchmark**. An agent receives a messy DataFrame, takes step-by-step structured cleaning actions, and is scored on how closely the cleaned output matches the hidden expected result.

This environment is:
- **Benchmark-ready** — deterministic graders, reproducible baseline, hidden expected outputs
- **Agent-compatible** — structured action space, typed observations, partial rewards
- **OpenEnv-compliant** — reset/step/state loop, typed models, YAML config
- **Production-flavored** — real business data patterns, not toy problems

---

## Architecture Overview

```
User / Agent
    │
    ▼
FastAPI Server  (server/app.py)
    │   /health  /tasks  /reset  /step  /state  /grader  /baseline
    ▼
CleanOpsEnvironment  (server/environment.py)
    │   reset(task_id) → Observation
    │   step(action)   → Observation + reward + done
    │   state()        → State snapshot
    ▼
Tasks           (tasks.py)          — messy + expected datasets
Cleaning Rules  (cleaning_rules.py) — pure helper functions
Metrics         (metrics.py)        — before/after quality measurement
Graders         (graders.py)        — deterministic 0.0→1.0 scoring
Agent           (agent.py)          — heuristic baseline agent
Baseline Runner (baseline.py)       — CLI: runs all 3 tasks
Client          (client.py)         — HTTP helper for local testing
```

### Key separation of concerns

| Component | Responsibility |
|---|---|
| `server/environment.py` | **Only place that mutates the DataFrame** |
| `graders.py` | Final scoring only (used post-episode, not during) |
| `cleaning_rules.py` | Pure functions — no state, no mutation |
| `agent.py` | Reads observations, returns actions — never touches DataFrame |
| `metrics.py` | Quality measurement — usable anywhere |

---

## Tasks

### Task 001 — EASY — Customer Contact Cleanup

**Dataset:** Customer CRM export (11 rows)

**Issues:**
- Bad capitalization (ALL CAPS names, lowercase cities)
- Extra whitespace in names and cities
- Invalid email addresses (malformed)
- Inconsistent phone number formats
- Duplicate customer rows (exact + near-duplicate)

**Goal:** Clean contact data to a consistent, validated format.

**Max steps:** 15

---

### Task 002 — MEDIUM — Sales Transaction Repair

**Dataset:** Sales/invoice table (11 rows)

**Issues:**
- Mixed date formats (ISO, DD/MM/YYYY, Month DD YYYY, etc.)
- Negative quantities
- Missing product categories
- Duplicate invoice rows
- Inconsistent currency formatting ($, usd, USD)
- Non-title-case product names

**Goal:** Repair transaction records while preserving valid data.

**Max steps:** 20

---

### Task 003 — HARD — Enterprise Ops Multi-Rule Cleanup

**Dataset:** Enterprise HR/ops reporting table (11 rows)

**Issues:**
- Missing required employee IDs
- Mixed date formats
- State/country cross-column mismatches (ON ≠ US)
- Incorrect total_comp calculations (salary × (1 + bonus_pct/100))
- Ambiguous duplicate records (same employee_id, different record_id)
- Bad capitalization in names and departments
- Extra whitespace

**Goal:** Clean under realistic enterprise business constraints.

**Max steps:** 25

---

## Action / Observation / State Design

### Action

```json
{
  "action_type": "standardize_format",
  "target_column": "name",
  "parameters": { "format": "title_case" }
}
```

**Available action types:**

| Action | Description |
|---|---|
| `inspect_column` | Read column stats (no mutation) |
| `standardize_format` | Strip/lower/upper/title_case a text column |
| `normalize_dates` | Parse and reformat date column to ISO |
| `validate_values` | Report invalid email/phone/date count |
| `replace_invalid` | Replace invalid values with null or a string |
| `fill_missing` | Fill nulls with value/mode/ffill/bfill |
| `remove_duplicates` | Drop exact duplicate rows |
| `merge_duplicates` | Deduplicate by a key column |
| `apply_business_rule` | Apply a named business rule |
| `finalize_cleaning` | End the episode |

**Business rules:** `fix_negative_quantities`, `fix_negative_prices`, `normalize_currency`, `fix_missing_employee_ids`, `fix_total_comp`, `fix_state_country`

---

### Observation

Returned after every `reset()` and `step()`:

```json
{
  "task_id": "task_001",
  "task_name": "Customer Contact Cleanup",
  "difficulty": "easy",
  "table_preview": [...],
  "schema": [{"name": "email", "dtype": "object", "null_count": 1, ...}],
  "quality_report": {
    "total_rows": 9,
    "duplicate_rows": 0,
    "missing_value_count": 1,
    "invalid_email_count": 0,
    "quality_score": 0.84
  },
  "allowed_actions": ["inspect_column", "standardize_format", ...],
  "step_count": 3,
  "max_steps": 15,
  "reward": 0.12,
  "done": false,
  "message": "Standardized 7 values in 'name' to title_case."
}
```

---

### State

Full internal state (includes hidden `expected_df` for grading):

```json
{
  "task_id": "task_001",
  "difficulty": "easy",
  "original_df": [...],
  "current_df": [...],
  "expected_df": [...],
  "issue_summary": {...},
  "action_history": [...],
  "step_count": 5,
  "max_steps": 15,
  "is_done": false,
  "current_score_estimate": 0.73
}
```

---

## Reward Logic

Rewards are computed **during each step** to provide partial progress feedback.

| Event | Reward |
|---|---|
| Inspect column | +0.01 (small, encourages informed decisions) |
| Validate values | +0.02 |
| Standardize text (N changes) | +0.05 + 0.02×N (capped at 10) + quality delta |
| Fill missing (N fills) | +0.05 + 0.02×N |
| Remove duplicates (N removed) | +0.10 + 0.02×N |
| Fix business rule | +0.08–0.12 |
| No-op (already clean) | −0.02 |
| Invalid action | −0.05 |
| Damage clean data | quality delta (negative) |
| Finalize — significant improvement | +0.20 |
| Finalize — minor improvement | +0.05 |
| Finalize — no improvement | −0.10 |
| Max steps exceeded without finalize | −0.10 |

Reward shaping is **separate** from final grading. The grader in `graders.py` compares `current_df` vs `expected_df` deterministically after the episode ends.

---

## Grader Design

Graders in `graders.py` compare the **cleaned output vs hidden expected output**, column by column.

Each task has its own grader with weighted column scores:

**Task 001 weights:**
- Row count: 15%
- Name (loose): 20%
- Email (strict): 20%
- Phone (strict): 20%
- City (loose): 15%
- Null handling: 10%

**Task 002 weights:**
- Row count: 10%, Date: 20%, Product: 15%, Category: 15%, Quantity: 15%, Currency: 10%, Total: 15%

**Task 003 weights:**
- Row count: 10%, Employee ID: 15%, Name: 10%, Department: 10%, Hire date: 15%, Country: 15%, Total comp: 15%, Null handling: 10%

Grader returns `score` [0.0–1.0] + `breakdown` dict + human-readable `message`.

---

## Before vs After Metrics

`metrics.py` computes before-and-after data quality for any DataFrame:

| Metric | Description |
|---|---|
| `missing_value_count` | Total null cells |
| `duplicate_row_count` | Exact duplicate rows |
| `invalid_email_count` | Invalid email format |
| `invalid_phone_count` | Unparseable phone numbers |
| `invalid_date_count` | Unparseable dates |
| `formatting_inconsistencies` | Leading/trailing spaces, case issues |
| `quality_score` | Composite 0–1 score |
| `preservation_ratio` | Fraction of original rows retained |
| `destructive_penalty` | Penalty if >40% rows removed |

---

## Setup

### Prerequisites
- Python 3.11+
- pip

### Install

```bash
# Clone or unzip the project
cd cleanops_openenv

# Install dependencies
pip install -r requirements.txt
```

---

## Local Run

```bash
# From the project root (cleanops_openenv/)
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Server starts at: http://localhost:7860

Interactive docs: http://localhost:7860/docs

---

## Run Baseline

```bash
# From cleanops_openenv/
python baseline.py
```

Output:
```
╔══════════════════════════════════════════════╗
║   CleanOps OpenEnv — Baseline Evaluation     ║
╚══════════════════════════════════════════════╝

  task_001  [EASY  ]  score: 0.7200  [██████████████░░░░░░]  steps: 10
             Quality: 0.612 → 0.841

  task_002  [MEDIUM]  score: 0.6500  [█████████████░░░░░░░]  steps: 14
             Quality: 0.573 → 0.792

  task_003  [HARD  ]  score: 0.5100  [██████████░░░░░░░░░░]  steps: 18
             Quality: 0.541 → 0.731

  ──────────────────────────────────────────────────
  Overall average score: 0.6267
```

---

## Docker Run

```bash
# Build (run from cleanops_openenv/ project root)
docker build -t cleanops-openenv -f server/Dockerfile .

# Run
docker run -p 7860:7860 cleanops-openenv

# With optional OpenAI key
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... cleanops-openenv
```

---

## Hugging Face Spaces Deployment

1. Create a new Space on https://huggingface.co/spaces
2. Choose **Docker** as the SDK
3. Upload all project files (or connect to GitHub)
4. HF Spaces auto-detects the Dockerfile and uses port 7860
5. The `openenv.yaml` provides metadata for the OpenEnv framework

The Dockerfile creates a non-root user (required by HF Spaces) and runs on port 7860.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server health check |
| GET | `/tasks` | List all tasks + action schema |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit a cleaning action |
| GET | `/state` | Full environment state |
| POST | `/grader` | Grade a cleaned dataset |
| POST | `/baseline` | Run baseline on all tasks |

---

## Sample API Calls

### Start episode
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_001"}'
```

### Submit action
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "standardize_format",
      "target_column": "name",
      "parameters": {"format": "title_case"}
    }
  }'
```

### Grade cleaned data
```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_001",
    "cleaned_rows": [
      {"customer_id": "C001", "name": "Alice Johnson", "email": "alice.johnson@example.com", ...}
    ]
  }'
```

### Run baseline
```bash
curl -X POST http://localhost:7860/baseline
```

---

## File Structure

```
cleanops_openenv/
├── models.py           ← Pydantic: Action, Observation, State
├── tasks.py            ← 3 tasks with messy + expected datasets
├── graders.py          ← Deterministic scoring 0.0→1.0
├── metrics.py          ← Before/after quality metrics
├── cleaning_rules.py   ← Pure cleaning helper functions
├── agent.py            ← Heuristic baseline agent (no API key needed)
├── baseline.py         ← CLI runner for all 3 tasks
├── client.py           ← HTTP client helper
├── openenv.yaml        ← OpenEnv framework config
├── requirements.txt
├── pyproject.toml
├── README.md
├── .env.example
└── server/
    ├── app.py          ← FastAPI application
    ├── environment.py  ← Core environment (only DataFrame mutator)
    └── Dockerfile
```

---

## Validator Readiness Checklist

- [x] Real-world task (not a game or toy)
- [x] Tabular/DataFrame based
- [x] Agentic step-by-step interaction
- [x] Typed Action, Observation, State models
- [x] `reset()`, `step()`, `state()` implemented
- [x] At least 3 tasks (easy, medium, hard)
- [x] Deterministic graders returning [0.0, 1.0]
- [x] Partial progress reward shaping
- [x] `/health`, `/tasks`, `/reset`, `/step`, `/state`, `/grader`, `/baseline` endpoints
- [x] `openenv.yaml` included
- [x] Working `Dockerfile` (HF Spaces compatible, port 7860)
- [x] Baseline runs with NO API key
- [x] Reproducible baseline scores
- [x] README included
- [x] `OPENAI_API_KEY` read from environment variable (optional)
