#!/usr/bin/env python3
"""
MCAR Hub Retention Scorer (OpenAI dual-judge, Colab-ready)
===========================================================
LLM-as-a-Judge using two independent GPT-4o-mini instances.

Colab Usage:
    from hub_scorer import score_hub_retention
    results = score_hub_retention(samples, predictions, api_key="sk-proj-...")
    print(f"HR: {results['HR']:.4f}")

Cost: ~600 samples x 2 judges x 1K tokens x $0.15/M = ~$0.09
"""

import json
import os
import asyncio
import aiohttp
import re
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm


API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"

HUB_JUDGE_PROMPT = """You are an expert evaluator assessing AI responses for logical comprehension quality.

TASK: Evaluate whether the model's answer correctly captures the core causal logic or main argument from the original document.

Original document excerpt (first 1500 chars):
{context_preview}

Reference answer (what a good answer should convey):
{reference_answer}

Model's answer to evaluate:
{prediction}

EVALUATION CRITERIA (score 1-5):
5 = Fully captures the core causal logic. The main argument, decision reason, or key insight is accurately conveyed. Equivalent quality to reference.
4 = Captures the main logic with minor omissions. The essence is correct but some nuance is missing.
3 = Roughly correct direction but with significant gaps. Shows partial understanding but misses key connections.
2 = Seriously misunderstands the core logic. The answer is wrong about the main causal chain or key reason.
1 = Completely irrelevant, nonsensical, or fails to address the question.

CRITICAL RULES:
- Do NOT penalize for missing specific numbers, names, or dates
- Do NOT reward for including irrelevant details or keyword stuffing
- Focus ONLY on whether the main causal chain / structural argument is correct
- The model may use different wording from the reference-that's fine if the logic is correct
- Be STRICT: most answers should be 2-4, not automatically 5

OUTPUT FORMAT: Respond with exactly one integer from 1 to 5, nothing else."""


async def _judge_one(session: aiohttp.ClientSession, headers: dict,
                     context_preview: str, reference: str, prediction: str) -> int:
    """Call GPT-4o-mini to judge a single prediction. Returns 1-5."""
    prompt = HUB_JUDGE_PROMPT.format(
        context_preview=context_preview[:1500],
        reference_answer=reference,
        prediction=prediction[:500]
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10
    }
    try:
        async with session.post(API_URL, headers=headers, json=payload, timeout=30) as resp:
            if resp.status != 200:
                print(f"  Judge API error {resp.status}")
                return 3
            result = await resp.json()
            text = result["choices"][0]["message"]["content"].strip()
            match = re.search(r'\b([1-5])\b', text)
            return int(match.group(1)) if match else 3
    except Exception as e:
        print(f"  Judge request error: {e}")
        return 3


async def _judge_sample(session, headers, sample, prediction, semaphore):
    """Judge one sample with two independent calls (dual judge)."""
    async with semaphore:
        # Extract context and reference
        context = ""
        reference = ""
        for turn in sample.get("turns", []):
            if turn.get("turn_type") == "context_injection":
                context = turn.get("content", "")
            if turn.get("turn_type") == "dual_probe":
                reference = turn.get("probe_b", {}).get("reference_answer", "")

        preview = context[:1500] if len(context) > 1500 else context

        # Two independent judges
        s1 = await _judge_one(session, headers, preview, reference, prediction)
        s2 = await _judge_one(session, headers, preview, reference, prediction)

        avg = (s1 + s2) / 2.0
        normalized = avg / 5.0

        return {
            "sample_id": sample.get("sample_id", ""),
            "domain": sample.get("domain", ""),
            "judge_scores": {"judge_1": s1, "judge_2": s2},
            "avg_raw": round(avg, 2),
            "normalized_score": round(normalized, 4),
            "disagreement": abs(s1 - s2)
        }


async def _run_batch(samples, predictions, api_key, max_concurrent):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [_judge_sample(session, headers, s, p, semaphore)
                 for s, p in zip(samples, predictions)]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Hub judging"):
            results.append(await coro)
        return results


def score_hub_retention(samples: List[Dict], predictions: List[str],
                        api_key: str = None, max_concurrent: int = 8) -> Dict:
    """
    Score Hub Retention for all samples using dual GPT-4o-mini judges.

    Args:
        samples: List of MCAR sample dicts (with reference_answer filled)
        predictions: List of model responses to Turn 5
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        max_concurrent: Max concurrent API calls (default 8)

    Returns:
        Dict with HR score, per-domain breakdown, and individual results
    """
    assert len(samples) == len(predictions), \
        f"Mismatch: {len(samples)} samples vs {len(predictions)} predictions"

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var.")

    print(f"Judging {len(samples)} samples (dual judge, concurrency={max_concurrent})...")

    try:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            _run_batch(samples, predictions, key, max_concurrent)
        )
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            _run_batch(samples, predictions, key, max_concurrent)
        )

    overall = sum(r["normalized_score"] for r in results) / len(results)
    domain_scores = defaultdict(list)
    for r in results:
        domain_scores[r["domain"]].append(r["normalized_score"])

    return {
        "HR": round(overall, 4),
        "by_domain": {d: round(sum(s)/len(s), 4) for d, s in domain_scores.items()},
        "individual": results
    }


# ── Direct execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLES_PATH = "/content/mcar-benchmark/data/final/mcar_600.jsonl"
    PREDS_PATH = "/content/predictions.jsonl"
    API_KEY = os.environ.get("OPENAI_API_KEY", "")

    with open(SAMPLES_PATH, "r") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    with open(PREDS_PATH, "r") as f:
        predictions = [json.loads(l).get("prediction", "") for l in f if l.strip()]

    r = score_hub_retention(samples, predictions, API_KEY)
    print(f"HR: {r['HR']:.4f}")
    for d, s in sorted(r["by_domain"].items()):
        print(f"  {d}: {s:.4f}")
