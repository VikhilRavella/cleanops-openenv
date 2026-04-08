"""
baseline.py — Baseline agent runner for CleanOps OpenEnv.

Runs the heuristic baseline agent on all 3 tasks.
Produces per-task scores and an overall average.

Usage:
    python baseline.py

Requirements:
    - NO external API key needed
    - Fully self-contained and reproducible
    - Safe to run for hackathon validation

Output format:
    task_001  EASY    score: 0.72
    task_002  MEDIUM  score: 0.65
    task_003  HARD    score: 0.51
    ─────────────────────────────
    Overall average:  0.629
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

from models import BaselineResult, BaselineResponse
from tasks import TASK_REGISTRY
from agent import get_agent
from graders import grade
from metrics import compute_before_after
from server.environment import CleanOpsEnvironment


def run_task(task_id: str) -> BaselineResult:
    """Run the heuristic agent on one task and return a scored result."""
    env   = CleanOpsEnvironment()
    obs   = env.reset(task_id)
    task  = TASK_REGISTRY[task_id]
    agent = get_agent(task_id, prefer_llm=False)

    # Plan from initial observation
    agent.plan(obs)

    steps_taken = 0
    while not obs.done and steps_taken < task.max_steps:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        steps_taken += 1

    # Final grading (separate from step rewards)
    grade_response = grade(task_id, env.current_df, env.expected_df)

    # Before / after metrics
    ba = compute_before_after(env._original_df, env.current_df)

    return BaselineResult(
        task_id=task_id,
        difficulty=task.difficulty,
        score=grade_response.score,
        steps_taken=steps_taken,
        metrics_before=ba["before"],
        metrics_after=ba["after"],
    )


def run_all_tasks() -> BaselineResponse:
    """Run baseline on all 3 tasks. Returns BaselineResponse."""
    results: List[BaselineResult] = []
    for task_id in sorted(TASK_REGISTRY.keys()):
        result = run_task(task_id)
        results.append(result)

    overall = round(sum(r.score for r in results) / len(results), 4)

    summary_lines = [f"  {r.task_id}  {r.difficulty.upper():<8}  score: {r.score:.4f}" for r in results]
    summary = "\n".join(summary_lines) + f"\n  Overall average: {overall:.4f}"

    return BaselineResponse(
        results=results,
        overall_average=overall,
        summary=summary,
    )


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   CleanOps OpenEnv — Baseline Evaluation     ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("Running heuristic baseline agent on all 3 tasks...\n")

    response = run_all_tasks()

    for result in response.results:
        bar = "█" * int(result.score * 20) + "░" * (20 - int(result.score * 20))
        print(f"  {result.task_id}  [{result.difficulty.upper():<6}]  score: {result.score:.4f}  [{bar}]  steps: {result.steps_taken}")
        print(f"           Quality: {result.metrics_before['quality_score']:.3f} → {result.metrics_after['quality_score']:.3f}")
        print()

    print(f"  {'─'*50}")
    print(f"  Overall average score: {response.overall_average:.4f}\n")


if __name__ == "__main__":
    main()
