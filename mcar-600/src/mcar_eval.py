#!/usr/bin/env python3
"""
MCAR-600 Complete Evaluation Suite (Colab-Ready, OpenAI-only)
===============================================================

All-in-one script for the MCAR-600 benchmark. Run directly in Google Colab.

QUICK START (in a Colab cell):
    # 1. Mount Drive and upload mcar_600.jsonl
    from google.colab import drive
    drive.mount('/content/drive')

    # 2. Install dependencies
    !pip install -q aiohttp openai tqdm

    # 3. Import and run evaluation
    import sys
    sys.path.append('/content/mcar-benchmark/src')
    import os
    os.environ['OPENAI_API_KEY'] = 'sk-proj-...'  # Your key here

    from mcar_eval import evaluate_predictions
    results = evaluate_predictions(
        samples_path='/content/drive/MyDrive/mcar_600.jsonl',
        predictions_path='/content/drive/MyDrive/predictions.jsonl',
        api_key='sk-proj-...'  # Same key
    )
    # Results printed automatically

COSTS:
    - Probe B answer generation: ~$0.05 (600 calls x 1K tokens)
    - Hub Retention judging: ~$0.09 (600 x 2 judges x 1K tokens)
    - Total per method: ~$0.14
"""

import json
import os
from typing import List, Dict

# ── Probe B Generation ───────────────────────────────────────────────────

JUDGE_PROMPT = """You are given a document excerpt and a question about its core logic.
Provide a concise, accurate answer (1-2 sentences) that captures the main causal chain or key insight.

Document excerpt (first 2000 chars):
{context}

Question: {question}

Requirements:
- Answer the core causal/logical question directly
- Be specific about WHO made WHAT decision and WHY
- 1-2 sentences maximum
- Focus on the structural logic, not surface details

Answer:"""


def _call_openai(messages, api_key, model="gpt-4o-mini", max_tokens=150, temperature=0.3):
    """Synchronous OpenAI API call."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        return ""


def generate_probe_b_answers(samples_path: str, output_path: str,
                              api_key: str = None, batch_size: int = 5):
    """
    Fill in reference_answer for Probe B using GPT-4o-mini.
    Processed in batches with progress printing.

    Args:
        samples_path: Path to mcar_600_raw.jsonl
        output_path: Path to save mcar_600.jsonl
        api_key: OpenAI API key
        batch_size: Save checkpoint every N samples
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key required")

    print(f"Loading samples from {samples_path}...")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} samples")

    print(f"\nGenerating Probe B reference answers...")
    filled = 0
    for i, sample in enumerate(samples):
        # Extract context and question
        context = ""
        question = ""
        for turn in sample.get("turns", []):
            if turn.get("turn_type") == "context_injection":
                context = turn.get("content", "")
            if turn.get("turn_type") == "dual_probe":
                content = turn.get("content", "")
                if "(B)" in content:
                    parts = content.split("(B)")
                    if len(parts) > 1:
                        question = parts[1].strip()

        if not context or not question:
            continue

        prompt = JUDGE_PROMPT.format(
            context=context[:2000],
            question=question
        )
        answer = _call_openai([{"role": "user", "content": prompt}], key)

        if answer:
            for turn in sample["turns"]:
                if turn.get("turn_type") == "dual_probe" and "probe_b" in turn:
                    turn["probe_b"]["reference_answer"] = answer
                    filled += 1

        if (i + 1) % batch_size == 0:
            print(f"  Progress: {i+1}/{len(samples)} ({filled} filled)")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=True) + "\n")

    print(f"\nFilled {filled}/{len(samples)} reference answers")
    print(f"Saved to {output_path}")
    return samples


# ── Island Survival Scoring ──────────────────────────────────────────────

def _island_score(pred: str, gt: str) -> int:
    if not pred or not gt:
        return 0
    return 1 if gt.lower() in pred.lower() else 0


def score_island_survival(samples: List[Dict], predictions: List[str]) -> Dict:
    assert len(samples) == len(predictions)
    results = []
    from collections import defaultdict
    domain_scores = defaultdict(list)

    for s, p in zip(samples, predictions):
        ia = s.get("island_metadata", {}).get("island_a", {})
        gt = ia.get("value", "")
        sc = _island_score(p, gt)
        results.append({
            "sample_id": s["sample_id"], "domain": s["domain"],
            "ground_truth": gt, "prediction": p[:200], "score": sc
        })
        domain_scores[s["domain"]].append(sc)

    overall = sum(r["score"] for r in results) / len(results)
    return {
        "IS": round(overall, 4),
        "by_domain": {d: round(sum(v)/len(v), 4) for d, v in domain_scores.items()},
        "individual": results
    }


# ── Hub Retention Scoring ────────────────────────────────────────────────

HUB_PROMPT = """You are an expert evaluator assessing AI responses for logical comprehension quality.

TASK: Evaluate whether the model's answer correctly captures the core causal logic or main argument from the original document.

Original document excerpt (first 1500 chars):
{context}

Reference answer (what a good answer should convey):
{reference}

Model's answer to evaluate:
{prediction}

EVALUATION CRITERIA (score 1-5):
5 = Fully captures the core causal logic. Equivalent quality to reference.
4 = Captures the main logic with minor omissions.
3 = Roughly correct direction but with significant gaps.
2 = Seriously misunderstands the core logic.
1 = Completely irrelevant or nonsensical.

CRITICAL RULES:
- Do NOT penalize for missing specific numbers, names, or dates
- Focus ONLY on whether the main causal chain is correct
- Be STRICT: most answers should be 2-4, not automatically 5

OUTPUT FORMAT: Respond with exactly one integer from 1 to 5, nothing else."""


def score_hub_retention(samples: List[Dict], predictions: List[str],
                        api_key: str = None, batch_size: int = 5) -> Dict:
    """
    Score Hub Retention with dual GPT-4o-mini judges.
    Processed synchronously in batches (Colab-friendly).
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key required")

    print(f"\nJudging Hub Retention (dual judge, {len(samples)} samples)...")
    results = []
    from collections import defaultdict
    domain_scores = defaultdict(list)

    for i, (s, p) in enumerate(zip(samples, predictions)):
        # Extract context and reference
        context = ""
        reference = ""
        for turn in s.get("turns", []):
            if turn.get("turn_type") == "context_injection":
                context = turn.get("content", "")
            if turn.get("turn_type") == "dual_probe":
                reference = turn.get("probe_b", {}).get("reference_answer", "")

        preview = context[:1500]
        pred = p[:500]

        # Judge 1
        prompt1 = HUB_PROMPT.format(context=preview, reference=reference, prediction=pred)
        r1 = _call_openai([{"role": "user", "content": prompt1}], key, max_tokens=10)

        # Judge 2 (independent call)
        r2 = _call_openai([{"role": "user", "content": prompt1}], key, max_tokens=10)

        # Parse scores
        import re
        s1 = int(re.search(r'\b([1-5])\b', r1).group(1)) if r1 and re.search(r'\b([1-5])\b', r1) else 3
        s2 = int(re.search(r'\b([1-5])\b', r2).group(1)) if r2 and re.search(r'\b([1-5])\b', r2) else 3

        avg = (s1 + s2) / 2.0
        norm = avg / 5.0

        results.append({
            "sample_id": s["sample_id"], "domain": s["domain"],
            "judge_scores": {"j1": s1, "j2": s2},
            "normalized_score": round(norm, 4)
        })
        domain_scores[s["domain"]].append(norm)

        if (i + 1) % batch_size == 0:
            print(f"  Judged: {i+1}/{len(samples)}")

    overall = sum(r["normalized_score"] for r in results) / len(results)
    return {
        "HR": round(overall, 4),
        "by_domain": {d: round(sum(v)/len(v), 4) for d, v in domain_scores.items()},
        "individual": results
    }


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate_scores(island_results: Dict, hub_results: Dict) -> Dict:
    """Aggregate IS and HR into PS and Delta."""
    is_list = island_results.get("individual", [])
    hr_list = hub_results.get("individual", [])
    hr_by_id = {r["sample_id"]: r for r in hr_list}

    per_sample = []
    for ir in is_list:
        sid = ir["sample_id"]
        is_val = ir["score"]
        hr_val = hr_by_id.get(sid, {}).get("normalized_score", 0.0)
        per_sample.append({
            "sample_id": sid, "domain": ir["domain"],
            "IS": is_val, "HR": round(hr_val, 4),
            "PS": round(is_val * hr_val, 4),
            "Delta": round(hr_val - is_val, 4)
        })

    n = len(is_list)
    is_overall = sum(ir["score"] for ir in is_list) / n
    hr_overall = sum(hr_by_id.get(ir["sample_id"], {}).get("normalized_score", 0) for ir in is_list) / n
    # PS = mean(IS_i * HR_i) per sample, NOT mean(IS) * mean(HR)
    # Paper: "Pareto Score IS × HR, averaged across all samples"
    ps_overall = sum(p["PS"] for p in per_sample) / n

    # Domain breakdown
    from collections import defaultdict
    d_scores = defaultdict(lambda: {"IS": [], "HR": []})
    for ps in per_sample:
        d_scores[ps["domain"]]["IS"].append(ps["IS"])
        d_scores[ps["domain"]]["HR"].append(ps["HR"])

    domain = {}
    for d, v in d_scores.items():
        i_s = sum(v["IS"]) / len(v["IS"])
        h_r = sum(v["HR"]) / len(v["HR"])
        domain[d] = {
            "IS": round(i_s, 4), "HR": round(h_r, 4),
            "PS": round(i_s * h_r, 4), "Delta": round(h_r - i_s, 4),
            "count": len(v["IS"])
        }

    return {
        "IS": round(is_overall, 4),
        "HR": round(hr_overall, 4),
        "PS": round(ps_overall, 4),
        "Delta": round(hr_overall - is_overall, 4),
        "by_domain": domain,
        "per_sample": per_sample
    }


def print_report(r: Dict):
    """Pretty-print MCAR results."""
    print("=" * 60)
    print("  MCAR-600 Evaluation Results")
    print("=" * 60)
    print(f"  IS (Island Survival):  {r['IS']:.4f}  ({r['IS']*100:.1f}%)")
    print(f"  HR (Hub Retention):    {r['HR']:.4f}  ({r['HR']*100:.1f}%)")
    print(f"  PS (Pareto Score):     {r['PS']:.4f}  ({r['PS']*100:.1f}%)")
    print(f"  Delta (Retention Bias): {r['Delta']:+.4f}")
    print()
    print("  By Domain:")
    print("  " + "-" * 56)
    for d, s in sorted(r["by_domain"].items(), key=lambda x: x[1]["PS"], reverse=True):
        print(f"    {d:>8}: IS={s['IS']:.3f} HR={s['HR']:.3f} PS={s['PS']:.3f} D={s['Delta']:+.3f} n={s['count']}")
    print("=" * 60)


# ── Main Evaluation Entry Point ──────────────────────────────────────────

def evaluate_predictions(samples_path: str, predictions_path: str,
                         api_key: str = None, output_json: str = None) -> Dict:
    """
    Complete MCAR evaluation pipeline.

    Args:
        samples_path: Path to mcar_600.jsonl (with reference_answer filled)
        predictions_path: Path to predictions.jsonl (each line: {"sample_id": "...", "prediction": "..."})
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        output_json: Optional path to save results JSON

    Returns:
        Dict with IS, HR, PS, Delta, by_domain, per_sample
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var.")

    # Load
    print(f"Loading samples: {samples_path}")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(samples)} samples")

    print(f"Loading predictions: {predictions_path}")
    with open(predictions_path, "r", encoding="utf-8") as f:
        pred_data = [json.loads(l) for l in f if l.strip()]
    # Match predictions to samples
    pred_by_id = {p.get("sample_id", ""): p.get("prediction", "") for p in pred_data}
    predictions = [pred_by_id.get(s["sample_id"], "") for s in samples]
    print(f"  {len([p for p in predictions if p])} matched predictions")

    # Score IS
    print("\n[1/3] Scoring Island Survival...")
    island = score_island_survival(samples, predictions)
    print(f"  IS: {island['IS']:.4f}")

    # Score HR
    print("\n[2/3] Scoring Hub Retention...")
    hub = score_hub_retention(samples, predictions, key)
    print(f"  HR: {hub['HR']:.4f}")

    # Aggregate
    print("\n[3/3] Aggregating...")
    results = aggregate_scores(island, hub)

    # Report
    print()
    print_report(results)

    # Save
    if output_json:
        import os
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "per_sample"}, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return results


# ── Direct execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("This script is designed to be imported in Colab.")
    print()
    print("Example usage:")
    print('  import sys; sys.path.append("/content/mcar-benchmark/src")')
    print("  from mcar_eval import evaluate_predictions")
    print("  results = evaluate_predictions(")
    print('      samples_path="/content/mcar_600.jsonl",')
    print('      predictions_path="/content/predictions.jsonl",')
    print('      api_key="sk-proj-...")
    print("  )")
