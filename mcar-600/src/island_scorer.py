#!/usr/bin/env python3
"""
MCAR Island Survival Scorer
============================
Exact substring matching for Probe A (Island Survival).

Colab Usage:
    from island_scorer import score_island_survival
    results = score_island_survival(samples, predictions)
    print(f"IS: {results['IS']:.4f}")
"""

import json
from typing import List, Dict
from collections import defaultdict


def _score_one(prediction: str, ground_truth: str) -> int:
    """Binary score: 1 if ground_truth found in prediction (case-insensitive), else 0."""
    if not prediction or not ground_truth:
        return 0
    return 1 if ground_truth.lower() in prediction.lower() else 0


def score_island_survival(samples: List[Dict], predictions: List[str]) -> Dict:
    """
    Score Island Survival for all samples.

    Args:
        samples: List of MCAR sample dicts (loaded from JSONL)
        predictions: List of model responses to Turn 5

    Returns:
        Dict with IS score, per-domain breakdown, and individual results
    """
    assert len(samples) == len(predictions), \
        f"Mismatch: {len(samples)} samples vs {len(predictions)} predictions"

    results = []
    domain_scores = defaultdict(list)

    for sample, pred in zip(samples, predictions):
        ia = sample.get("island_metadata", {}).get("island_a", {})
        gt = ia.get("value", "")
        score = _score_one(pred, gt)

        r = {
            "sample_id": sample.get("sample_id", ""),
            "domain": sample.get("domain", ""),
            "island_type": ia.get("type", ""),
            "ground_truth": gt,
            "prediction": pred[:200],
            "score": score
        }
        results.append(r)
        domain_scores[sample.get("domain", "")].append(score)

    overall = sum(r["score"] for r in results) / len(results) if results else 0

    return {
        "IS": round(overall, 4),
        "by_domain": {d: round(sum(s)/len(s), 4) for d, s in domain_scores.items()},
        "individual": results
    }


# ── Direct execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: load samples and predictions, then score
    SAMPLES_PATH = "/content/mcar-benchmark/data/final/mcar_600.jsonl"
    PREDS_PATH = "/content/predictions.jsonl"

    with open(SAMPLES_PATH, "r") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    with open(PREDS_PATH, "r") as f:
        predictions = [json.loads(l).get("prediction", "") for l in f if l.strip()]

    r = score_island_survival(samples, predictions)
    print(f"IS: {r['IS']:.4f}")
    for d, s in sorted(r["by_domain"].items()):
        print(f"  {d}: {s:.4f}")
