"""
client.py — HTTP client helper for CleanOps OpenEnv.

Wraps the FastAPI server endpoints for easy local testing
and baseline execution from the command line.

Usage:
    from client import CleanOpsClient

    client = CleanOpsClient("http://localhost:7860")
    obs = client.reset("task_001")
    obs = client.step({"action_type": "remove_duplicates", "parameters": {"keep": "first"}})
    state = client.state()
    result = client.grade("task_001", cleaned_rows)
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error


class CleanOpsClient:
    """
    Lightweight HTTP client for the CleanOps OpenEnv server.
    Uses only stdlib (no requests dependency).
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: Dict) -> Dict:
        url  = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} from {path}: {body}") from e

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} from {path}: {body}") from e

    # ── Endpoints ──────────────────────────────────────────────────────────────

    def health(self) -> Dict:
        """GET /health"""
        return self._get("/health")

    def tasks(self) -> List[Dict]:
        """GET /tasks"""
        return self._get("/tasks")

    def reset(self, task_id: str) -> Dict:
        """POST /reset — start a new episode."""
        return self._post("/reset", {"task_id": task_id})

    def step(self, action: Dict) -> Dict:
        """
        POST /step — submit a cleaning action.
        action example:
            {"action_type": "remove_duplicates", "parameters": {"keep": "first"}}
        """
        return self._post("/step", {"action": action})

    def state(self) -> Dict:
        """GET /state — full environment state."""
        return self._get("/state")

    def grade(self, task_id: str, cleaned_rows: List[Dict]) -> Dict:
        """POST /grader — grade a cleaned dataset."""
        return self._post("/grader", {"task_id": task_id, "cleaned_rows": cleaned_rows})

    def baseline(self) -> Dict:
        """POST /baseline — run baseline agent on all tasks."""
        return self._post("/baseline", {})

    # ── Convenience helpers ────────────────────────────────────────────────────

    def run_episode(self, task_id: str, actions: List[Dict]) -> Dict:
        """
        Run a full episode: reset + submit all actions + return final state.
        Returns final observation dict.
        """
        obs = self.reset(task_id)
        print(f"[reset] task={task_id}  quality={obs['quality_report']['quality_score']:.3f}")
        for i, action in enumerate(actions, 1):
            obs = self.step(action)
            print(f"[step {i}] {action['action_type']}  reward={obs['reward']:.3f}  done={obs['done']}")
            if obs["done"]:
                break
        return obs

    def print_quality_report(self, state: Optional[Dict] = None) -> None:
        """Print a human-readable quality summary from state or current state."""
        if state is None:
            state = self.state()
        metrics = state.get("issue_summary", {})
        print("\n── Quality Report ──────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<30} {v}")
        print("────────────────────────────────────────────\n")


# ─── Quick CLI demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    client = CleanOpsClient(base)

    print("── Health ──")
    print(client.health())

    print("\n── Tasks ──")
    for t in client.tasks():
        print(f"  {t['task_id']}  [{t['difficulty']}]  {t['task_name']}")

    print("\n── Running quick episode on task_001 ──")
    actions = [
        {"action_type": "remove_duplicates",   "parameters": {"keep": "first"}},
        {"action_type": "standardize_format",  "target_column": "name", "parameters": {"format": "title_case"}},
        {"action_type": "standardize_format",  "target_column": "email", "parameters": {"format": "lower"}},
        {"action_type": "normalize_dates",     "target_column": "date", "parameters": {"output_format": "%Y-%m-%d"}} if False else None,
        {"action_type": "finalize_cleaning",   "parameters": {}},
    ]
    actions = [a for a in actions if a is not None]

    final_obs = client.run_episode("task_001", actions)
    print(f"\nFinal quality score: {final_obs['quality_report']['quality_score']:.3f}")

    print("\n── Grading ──")
    state = client.state()
    result = client.grade("task_001", state["current_df"])
    print(f"  Score: {result['score']:.4f}")
    print(f"  Breakdown: {result['breakdown']}")
    print(f"  Message: {result['message']}")
