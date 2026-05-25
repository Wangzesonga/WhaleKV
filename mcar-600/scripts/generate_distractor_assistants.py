"""
generate_distractor_assistants.py

Regenerate MCAR-600 distractor reference_assistants (turns 2-4) using any
HuggingFace causal LM. By default the dataset ships with Claude Sonnet 4.6
generated assistants; use this script to produce a Llama or other variant.

Usage:
    python scripts/generate_distractor_assistants.py \
        --model  meta-llama/Llama-3.1-8B-Instruct \
        --input  data/mcar_600_merged.jsonl \
        --output data/mcar_600_merged_llama_distractors.jsonl \
        --device cuda
"""

import argparse
import json
import copy
from datetime import date
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, question: str, device: str, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--input", default="data/mcar_600_merged.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    with open(args.input) as f:
        samples = [json.loads(l) for l in f]

    # Deduplicate distractor questions to avoid redundant generation
    unique_questions = {}
    for s in samples:
        for turn in s["turns"]:
            if "distractor" in turn.get("turn_type", ""):
                q = turn["content"]
                if q not in unique_questions:
                    unique_questions[q] = None

    print(f"Generating answers for {len(unique_questions)} unique distractor questions...")
    for q in tqdm(unique_questions):
        unique_questions[q] = generate_response(
            model, tokenizer, q, args.device, args.max_new_tokens
        )

    # Rebuild samples with new distractor answers
    updated = []
    for s in samples:
        ns = copy.deepcopy(s)
        ra = ns.get("reference_assistants", {})
        for turn in ns["turns"]:
            ti = turn["turn_idx"]
            if "distractor" in turn.get("turn_type", ""):
                q = turn["content"]
                ra[f"turn_{ti}"] = unique_questions[q]

        ns["reference_assistants"] = ra
        ns.setdefault("inference_spec", {})
        ns["inference_spec"]["distractor_generation"] = {
            "generator_model": args.model,
            "generation_date": str(date.today()),
            "gen_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        }
        updated.append(ns)

    with open(args.output, "w") as f:
        for s in updated:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(updated)} samples to {args.output}")


if __name__ == "__main__":
    main()
