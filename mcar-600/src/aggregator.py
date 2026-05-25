#!/usr/bin/env python3
"""
MCAR Score Aggregator (Colab-ready)
====================================
Aggregates IS and HR into PS (Pareto Score) and Delta.

Colab Usage:
    from aggregator import aggregate_mcar_scores
    results = aggregate_mcar_scores(island_results, hub_results)
    print(f"PS: {results['PS']:.4f}, Delta: {results['Delta']:+.4f}")
"""

from typing import Dict, List


def aggregate_mcar_scores(island_results: Dict, hub_results: Dict) -> Dict:
    """
    Aggregate IS and HR into unified MCAR metrics.

    Args:
        island_results: Output from score_island_survival()
        hub_results: Output from score_hub_retention()

    Returns:
        Dict with IS, HR, PS, Delta, and per-sample breakdown
    """
    is_scores = island_results.get("individual", [])
    hr_scores = hub_results.get("individual", [])

    # Build lookup by sample_id
    hr_by_id = {r["sample_id"]: r for r in hr_scores}

    per_sample = []
    total_is = 0
    total_hr = 0

    for is_r in is_scores:
        sid = is_r["sample_id"]
        domain = is_r.get("domain", "")
        is_val = is_r["score"]  # 0 or 1

        hr_r = hr_by_id.get(sid, {})
        hr_val = hr_r.get("normalized_score", 0.0)

        ps = is_val * hr_val
        delta = hr_val - is_val

        per_sample.append({
            "sample_id": sid,
            "domain": domain,
            "IS": is_val,
            "HR": round(hr_val, 4),
            "PS": round(ps, 4),
            "Delta": round(delta, 4)
        })

        total_is += is_val
        total_hr += hr_val

    n = len(is_scores)
    overall_is = total_is / n if n else 0
    overall_hr = total_hr / n if n else 0
    # PS = mean(IS_i * HR_i) per sample, NOT mean(IS) * mean(HR)
    # Paper: "Pareto Score IS × HR, averaged across all samples"
    overall_ps = sum(p["PS"] for p in per_sample) / n if n else 0
    overall_delta = overall_hr - overall_is

    # Domain breakdowns
    domains = set()
    for r in is_scores:
        domains.add(r.get("domain", ""))

    domain_breakdown = {}
    for d in sorted(domains):
        d_samples = [p for p in per_sample if p["domain"] == d]
        if d_samples:
            domain_breakdown[d] = {
                "IS": round(sum(p["IS"] for p in d_samples) / len(d_samples), 4),
                "HR": round(sum(p["HR"] for p in d_samples) / len(d_samples), 4),
                "PS": round(sum(p["PS"] for p in d_samples) / len(d_samples), 4),
                "Delta": round(sum(p["Delta"] for p in d_samples) / len(d_samples), 4),
                "count": len(d_samples)
            }

    return {
        "IS": round(overall_is, 4),
        "HR": round(overall_hr, 4),
        "PS": round(overall_ps, 4),
        "Delta": round(overall_delta, 4),
        "by_domain": domain_breakdown,
        "per_sample": per_sample,
        "interpretation": {
            "PS > 0.70": "Excellent: both dimensions strong",
            "PS 0.50-0.70": "Good: balanced performance",
            "PS 0.20-0.50": "Poor: one or both dimensions weak",
            "PS < 0.20": "Failed: severe compression-induced degradation",
            "Delta > +0.1": "Global-biased (better at logic than precise retrieval)",
            "Delta < -0.1": "Precision-biased (better at precise retrieval than logic)",
            "Delta ~ 0": "Balanced retention (ideal)"
        }
    }


def print_mcar_report(results: Dict):
    """Pretty-print MCAR results."""
    print("=" * 55)
    print("MCAR-600 Evaluation Results")
    print("=" * 55)
    print()
    print(f"  IS (Island Survival):  {results['IS']:.4f}  ({results['IS']*100:.1f}%)")
    print(f"  HR (Hub Retention):    {results['HR']:.4f}  ({results['HR']*100:.1f}%)")
    print(f"  PS (Pareto Score):     {results['PS']:.4f}  ({results['PS']*100:.1f}%)")
    print(f"  Delta (Retention Bias): {results['Delta']:+.4f}")
    print()
    print("By Domain:")
    print("-" * 55)
    for domain, scores in sorted(results["by_domain"].items(),
                                  key=lambda x: x[1]["PS"], reverse=True):
        print(f"  {domain:>8}: IS={scores['IS']:.3f}  HR={scores['HR']:.3f}  "
              f"PS={scores['PS']:.3f}  Delta={scores['Delta']:+.3f}  (n={scores['count']})")
    print()
    print("=" * 55)
