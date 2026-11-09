#!/usr/bin/env python3
"""Acceptance record verification runner across all 40 plan tasks."""

from __future__ import annotations

from pathlib import Path
import re
import sys

PLAN_PATH = Path("docs/plan.md")
ACCEPTANCE_DIR = Path("docs/acceptance")
TASK_HEADER_PATTERN = re.compile(r"^####\s+(T\d+\.\d+)\s+(.+)$")
STATUS_PASS_PATTERN = re.compile(r"(?im)^\s*(?:#{1,6}\s+)?[-*]?\s*[*_]*status[*_:]*\s*:?\s*pass\b")


def parse_plan_tasks(plan_path: Path = PLAN_PATH) -> list[tuple[str, str]]:
    """Parse (task_id, title) tuples from docs/plan.md."""
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    tasks: list[tuple[str, str]] = []
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        match = TASK_HEADER_PATTERN.match(line.strip())
        if match:
            tasks.append((match.group(1), match.group(2).strip()))
    return tasks


def check_acceptance_records(
    tasks: list[tuple[str, str]], acceptance_dir: Path = ACCEPTANCE_DIR
) -> tuple[int, int, int, list[tuple[str, str, str]]]:
    """Check status of each task acceptance document."""
    passed = 0
    failed = 0
    missing = 0
    records: list[tuple[str, str, str]] = []

    for task_id, title in tasks:
        doc_path = acceptance_dir / f"{task_id}.md"
        if not doc_path.is_file():
            missing += 1
            records.append((task_id, title, "MISSING"))
            continue

        content = doc_path.read_text(encoding="utf-8")
        if STATUS_PASS_PATTERN.search(content):
            passed += 1
            records.append((task_id, title, "PASS"))
        else:
            failed += 1
            records.append((task_id, title, "FAIL"))

    return passed, failed, missing, records


def main() -> int:
    tasks = parse_plan_tasks()
    passed, failed, missing, records = check_acceptance_records(tasks)

    print("=" * 80)
    print(f"{'Task':<8} | {'Status':<8} | {'Title'}")
    print("-" * 80)
    for task_id, title, status in records:
        print(f"{task_id:<8} | {status:<8} | {title}")
    print("=" * 80)
    print(f"Summary: {len(tasks)} tasks parsed | Passed: {passed} | Failed: {failed} | Missing: {missing}")

    if len(tasks) == 40 and passed == 40 and failed == 0 and missing == 0:
        print("Acceptance Gate Result: 40/40 PASS")
        return 0

    print("Acceptance Gate Result: INCOMPLETE or FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
