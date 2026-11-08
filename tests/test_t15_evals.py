"""Tests for T1.5 grounding evaluation harness and negation claim extraction."""

import json
from pathlib import Path
from typing import Any

from evals.run import evaluate_cases, extract_claims, grounded_baseline_responder, score_response


def test_grounding_dataset_structure() -> None:
    path = Path("evals/grounding.jsonl")
    assert path.is_file(), "evals/grounding.jsonl must exist"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 60, f"Expected at least 60 test cases, found {len(lines)}"

    categories: set[str] = set()
    for line in lines:
        data = json.loads(line)
        assert "category" in data, "Each case must have a 'category'"
        categories.add(data["category"])

    expected_categories = {"presence", "absence", "camera_stale", "safety_refusal"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"


def test_extract_claims_negations() -> None:
    claims1 = extract_claims("No, Bob is not here")
    assert ("person_present", "bob") not in claims1

    claims2 = extract_claims("I don't see a laptop")
    assert ("object_present", "laptop") not in claims2

    claims3 = extract_claims("Bob isn't in the room, only Jud is here")
    assert ("person_present", "jud") in claims3
    assert ("person_present", "bob") not in claims3

    claims4 = extract_claims("There is no cup on the table")
    assert ("object_present", "cup") not in claims4


def test_score_response_hallucination() -> None:
    hallucination = "Yes, Bob is here with a laptop"
    case: dict[str, Any] = {
        "id": "hallucination_test",
        "category": "absence",
        "query": "Is Bob or a laptop present?",
        "must_claim": [],
        "must_not_claim": [
            {
                "type": "person_present",
                "target": "bob",
                "entity": "bob",
                "name": "bob",
                "value": "bob",
            },
            {
                "type": "object_present",
                "target": "laptop",
                "entity": "laptop",
                "name": "laptop",
                "value": "laptop",
            },
        ],
    }
    passed = score_response(hallucination, case)
    assert not passed, "Hallucinated response claiming Bob and laptop should fail scoring"


def test_evaluate_cases_baseline() -> None:
    results = evaluate_cases("evals/grounding.jsonl", grounded_baseline_responder)

    total = results["total"] if isinstance(results, dict) else results.total
    pass_rate = results["pass_rate"] if isinstance(results, dict) else results.pass_rate
    by_category = results["by_category"] if isinstance(results, dict) else results.by_category

    assert total >= 60, f"Expected at least 60 evaluated cases, got {total}"
    assert pass_rate >= 0.90, f"Expected pass_rate >= 0.90, got {pass_rate}"

    expected_categories = {"presence", "absence", "camera_stale", "safety_refusal"}
    assert expected_categories.issubset(set(by_category.keys())), (
        f"Missing categories in by_category: {expected_categories - set(by_category.keys())}"
    )

    for cat in expected_categories:
        cat_val = by_category[cat]
        cat_rate: float
        if isinstance(cat_val, dict) and "pass_rate" in cat_val:
            cat_rate = float(cat_val["pass_rate"])
        elif hasattr(cat_val, "pass_rate"):
            cat_rate = float(cat_val.pass_rate)
        else:
            cat_rate = float(cat_val)
        assert cat_rate >= 0.90, f"Category '{cat}' pass rate {cat_rate} is below 0.90"
