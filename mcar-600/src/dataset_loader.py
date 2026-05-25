#!/usr/bin/env python3
"""
MCAR-600 Dataset Loader
=======================
Unified loader for MCAR-600 (split into domain files).

Usage:
    from dataset_loader import load_mcar_dataset
    
    # Load all 600 samples
    samples = load_mcar_dataset("data/final")
    
    # Load single domain
    narr_samples = load_mcar_dataset("data/final", domain="narr")  # 200
    meet_samples = load_mcar_dataset("data/final", domain="meet")  # 200
    code_samples = load_mcar_dataset("data/final", domain="code")  # 200
    
    # Load from merged file (if available)
    samples = load_mcar_dataset("data/final/mcar_600_merged.jsonl")
"""

import json
import os
from typing import List, Dict, Optional


def load_mcar_dataset(data_dir: str, domain: Optional[str] = None) -> List[Dict]:
    """
    Load MCAR-600 dataset.
    
    Args:
        data_dir: Directory containing mcar_600_*.jsonl files,
                  or path to a single merged .jsonl file
        domain: "narr", "meet", "code", or None for all
    
    Returns:
        List of sample dicts
    """
    # Check if data_dir is a single file
    if os.path.isfile(data_dir):
        with open(data_dir, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    
    # Load from domain-split files
    if domain:
        filepath = os.path.join(data_dir, f"mcar_600_{domain}.jsonl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    
    # Load all domains
    all_samples = []
    for d in ["narr", "meet", "code"]:
        filepath = os.path.join(data_dir, f"mcar_600_{d}.jsonl")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                samples = [json.loads(l) for l in f if l.strip()]
                all_samples.extend(samples)
    
    # Sort by sample_id for consistent ordering
    all_samples.sort(key=lambda s: s.get("sample_id", ""))
    return all_samples


def get_statistics(samples: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    from collections import Counter, defaultdict
    
    domains = Counter(s["domain"] for s in samples)
    
    word_counts = [s["context_length_words"] for s in samples]
    token_counts = [s["context_length_estimate_tokens"] for s in samples]
    
    island_a_types = Counter(s["island_metadata"]["island_a"]["type"] for s in samples)
    island_b_types = Counter(s["island_metadata"]["island_b"]["type"] for s in samples)
    
    # Check Probe B answer status
    has_answer = sum(1 for s in samples 
                     if s["turns"][5].get("probe_b", {}).get("reference_answer"))
    
    return {
        "total_samples": len(samples),
        "by_domain": dict(domains),
        "context_length_words": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(sum(word_counts) / len(word_counts)),
            "median": sorted(word_counts)[len(word_counts) // 2]
        },
        "context_length_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts)),
            "median": sorted(token_counts)[len(token_counts) // 2]
        },
        "island_a_types": dict(island_a_types),
        "island_b_types": dict(island_b_types),
        "probe_b_answered": f"{has_answer}/{len(samples)}"
    }


def merge_domain_files(data_dir: str, output_path: str):
    """Merge domain-split files into a single file."""
    samples = load_mcar_dataset(data_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=True) + "\n")
    print(f"Merged {len(samples)} samples to {output_path}")


# ── Direct execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/final"
    
    print("Loading MCAR-600 dataset...")
    samples = load_mcar_dataset(data_dir)
    stats = get_statistics(samples)
    
    print(f"\n{'='*50}")
    print(f"MCAR-600 Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"By domain: {stats['by_domain']}")
    print(f"\nContext length (words):")
    for k, v in stats['context_length_words'].items():
        print(f"  {k}: {v}")
    print(f"\nContext length (estimated tokens):")
    for k, v in stats['context_length_tokens'].items():
        print(f"  {k}: {v}")
    print(f"\nIsland A types: {len(stats['island_a_types'])} types")
    print(f"Island B types: {len(stats['island_b_types'])} types")
    print(f"Probe B answered: {stats['probe_b_answered']}")
