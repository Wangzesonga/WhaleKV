#!/usr/bin/env python3
"""
MCAR-Standard Reference Implementation
======================================

Implements the MCAR-Standard protocol: turn-by-turn inference with true
KV cache reuse across turns, using `past_key_values` to avoid re-encoding
the full history on every turn.

KVPress integration:
    Use `with press(model):` BEFORE calling run_mcar_standard. The press
    registers hooks on the model's attention layers and fires automatically
    during model.generate(). Do NOT pass a press into run_mcar_standard;
    instead wrap the entire call:

        press = WhaleKVAdaptivePress(compression_ratio=0.5)
        with press(model):
            results = run_mcar_standard(model, tokenizer, samples)

    This is the only correct way to use KVPress with MCAR.

MCAR-Standard protocol:
    Step 1: Prefill Turn 0 (long document) → capture past_key_values
    Step 2: For T in [1,2,3,4]:
              - Append user query to KV
              - Generate assistant (max_new_tokens=128, temp=0.0)
                OR inject reference_assistant text into KV
              - KV cache now contains full history up to T
    Step 3: Append Turn 5 (dual-probe) → generate final answer
              (max_new_tokens=256, temp=0.1)
"""

import json
import os
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


TURN14_CONFIG = {"max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0}
TURN5_CONFIG  = {"max_new_tokens": 256, "temperature": 0.1, "top_p": 1.0}


def _encode_new_text(
    tokenizer: PreTrainedTokenizer,
    text: str,
    role: str,
    add_generation_prompt: bool,
    device: str,
) -> torch.Tensor:
    """Tokenize a single new turn without re-encoding history."""
    # Build a minimal single-turn chat to get the correct role wrapper tokens
    msg = [{"role": role, "content": text}]
    ids = tokenizer.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    ).to(device)
    # Strip the BOS token if present (it is already in past_key_values from turn 0)
    if ids[0, 0] == tokenizer.bos_token_id:
        ids = ids[:, 1:]
    return ids


def _generate_with_cache(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    past_key_values,
    gen_config: Dict,
) -> tuple:
    """
    Generate new tokens given input_ids and existing past_key_values.
    Returns (generated_text_ids, updated_past_key_values).
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            past_key_values=past_key_values,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["temperature"] > 0,
            use_cache=True,
            return_dict_in_generate=True,
        )
    # outputs.sequences includes the input_ids prefix; extract only new tokens
    new_ids = outputs.sequences[:, input_ids.shape[1]:]
    return new_ids, outputs.past_key_values


def run_mcar_standard(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: List[Dict],
    device: str = "cuda",
    use_reference_assistant: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    MCAR-Standard: true KV cache reuse across turns via past_key_values.

    KVPress MUST be applied as a context manager before calling this function:
        with press(model):
            results = run_mcar_standard(model, tokenizer, samples)

    Args:
        model: HuggingFace causal LM (already on device, bfloat16 recommended)
        tokenizer: Matching tokenizer
        samples: List of MCAR-600 sample dicts
        device: "cuda" or "cpu"
        use_reference_assistant: If True, inject reference_assistants for T1-4
                                  (required for leaderboard submissions)
        verbose: Print progress

    Returns:
        Dict with predictions (matched by sample_id), transcripts, protocol info
    """
    model.eval()
    predictions: List[Dict] = []
    transcripts: List[Dict] = []

    if verbose:
        print(f"MCAR-Standard: {len(samples)} samples")
        print(f"  use_reference_assistant: {use_reference_assistant}")
        print(f"  Compression: applied via model hooks (if any)")

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}")

        turns = sample["turns"]
        turn0_content = turns[0]["content"]
        user_contents = [turns[t]["content"] for t in range(1, 5)]
        turn5_content = turns[5]["content"]

        ref = sample.get("reference_assistants", {}) if use_reference_assistant else {}

        # ── Step 1: Prefill Turn 0 (long document) ───────────────────────────
        t0_messages = [{"role": "user", "content": turn0_content}]
        t0_ids = tokenizer.apply_chat_template(
            t0_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            t0_out = model(input_ids=t0_ids, use_cache=True)
        past_kv = t0_out.past_key_values
        transcript: Dict[str, str] = {}

        # ── Step 2: Turns 1-4 ────────────────────────────────────────────────
        for t_idx in range(1, 5):
            # Append user query to KV
            user_ids = _encode_new_text(
                tokenizer, user_contents[t_idx - 1], "user",
                add_generation_prompt=True, device=device
            )
            with torch.no_grad():
                user_out = model(input_ids=user_ids, past_key_values=past_kv, use_cache=True)
            past_kv = user_out.past_key_values

            ref_text = ref.get(f"turn_{t_idx}", "")
            if ref_text:
                # Inject reference assistant text into KV (fixed transcript mode)
                asst_ids = _encode_new_text(
                    tokenizer, ref_text, "assistant",
                    add_generation_prompt=False, device=device
                )
                with torch.no_grad():
                    asst_out = model(input_ids=asst_ids, past_key_values=past_kv, use_cache=True)
                past_kv = asst_out.past_key_values
                assistant_text = ref_text
            else:
                # Self-generate assistant response
                # The generation prompt was already appended via add_generation_prompt=True above
                dummy_ids = torch.zeros((1, 0), dtype=torch.long, device=device)
                new_ids, past_kv = _generate_with_cache(model, dummy_ids, past_kv, TURN14_CONFIG)
                assistant_text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

            transcript[f"turn_{t_idx}"] = assistant_text

        # ── Step 3: Turn 5 dual-probe ─────────────────────────────────────────
        probe_ids = _encode_new_text(
            tokenizer, turn5_content, "user",
            add_generation_prompt=True, device=device
        )
        with torch.no_grad():
            probe_out = model(input_ids=probe_ids, past_key_values=past_kv, use_cache=True)
        probe_kv = probe_out.past_key_values

        dummy_ids = torch.zeros((1, 0), dtype=torch.long, device=device)
        final_ids, _ = _generate_with_cache(model, dummy_ids, probe_kv, TURN5_CONFIG)
        final_answer = tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()

        predictions.append({"sample_id": sid, "prediction": final_answer})
        transcripts.append({"sample_id": sid, "transcript": transcript})

    return {
        "protocol": "standard",
        "predictions": predictions,
        "transcripts": transcripts,
        "use_reference_assistant": use_reference_assistant,
    }


def run_mcar_oneshot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: List[Dict],
    device: str = "cuda",
) -> Dict:
    """
    MCAR-OneShot (ablation only): concatenate all turns, single generate.

    FORBIDDEN as a main leaderboard result. Use only to quantify the gap
    between true multi-turn KV reuse vs simple prompt lengthening.
    """
    model.eval()
    predictions = []

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        # Include all user turns (0-5) as a single concatenated prompt
        messages = [{"role": "user", "content": t["content"]}
                    for t in sample["turns"] if t["role"] == "user"]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=TURN5_CONFIG["max_new_tokens"],
                temperature=TURN5_CONFIG["temperature"],
                top_p=TURN5_CONFIG["top_p"],
                do_sample=TURN5_CONFIG["temperature"] > 0,
                use_cache=True,
            )
        pred = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        predictions.append({"sample_id": sid, "prediction": pred})

        if (i + 1) % 50 == 0:
            print(f"  OneShot {i + 1}/{len(samples)}")

    return {
        "protocol": "oneshot",
        "predictions": predictions,
        "transcripts": [],
        "use_reference_assistant": False,
    }


def save_mcar_results(results: Dict, output_dir: str, method_name: str):
    """
    Save MCAR run results in the standard submission format.

    Predictions JSONL format (required by evaluate_predictions):
        {"sample_id": "MCAR-NARR-000", "prediction": "..."}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predictions (matched by sample_id)
    pred_path = os.path.join(output_dir, f"{method_name}_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in results["predictions"]:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Transcripts (optional, for inspection)
    if results.get("transcripts"):
        trans_path = os.path.join(output_dir, f"{method_name}_transcripts.jsonl")
        with open(trans_path, "w", encoding="utf-8") as f:
            for row in results["transcripts"]:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Protocol metadata
    proto_path = os.path.join(output_dir, f"{method_name}_protocol.json")
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": results["protocol"],
                "use_reference_assistant": results.get("use_reference_assistant", False),
                "turn14_config": TURN14_CONFIG,
                "turn5_config": TURN5_CONFIG,
            },
            f,
            indent=2,
        )

    print(f"Saved: {pred_path}")


def compute_degradation(
    results_by_ratio: Dict[float, Dict]
) -> Dict:
    """
    Compute the Degradation metric from Table 10: PS drop from r=0.2 to r=0.8.

    Args:
        results_by_ratio: {compression_ratio: evaluate_predictions_output}
            e.g. {0.2: results_r02, 0.5: results_r05, 0.8: results_r08}

    Returns:
        Dict with PS per ratio and Degradation = PS(r=0.2) - PS(r=0.8)

    Example:
        >>> r = compute_degradation({
        ...     0.2: evaluate_predictions(..., compression_ratio=0.2),
        ...     0.5: evaluate_predictions(..., compression_ratio=0.5),
        ...     0.8: evaluate_predictions(..., compression_ratio=0.8),
        ... })
        >>> print(r)
        {'PS_r0.2': 88.7, 'PS_r0.5': 78.9, 'PS_r0.8': 66.5, 'Degradation': -22.2}
    """
    out = {}
    for ratio, res in sorted(results_by_ratio.items()):
        key = f"PS_r{ratio:.1f}".replace(".", "")
        out[key] = round(res["PS"] * 100, 1)  # convert to percentage

    ratios = sorted(results_by_ratio.keys())
    if ratios[0] in results_by_ratio and ratios[-1] in results_by_ratio:
        ps_low  = results_by_ratio[ratios[0]]["PS"] * 100
        ps_high = results_by_ratio[ratios[-1]]["PS"] * 100
        out["Degradation"] = round(ps_high - ps_low, 1)   # negative = degradation

    return out


if __name__ == "__main__":
    print("Import run_mcar_standard / run_mcar_oneshot / compute_degradation from this module.")
    print()
    print("KVPress usage:")
    print("  with press(model):")
    print("      results = run_mcar_standard(model, tokenizer, samples)")
